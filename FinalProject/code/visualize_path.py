#!/usr/bin/env python3
# =======================================================================================
# visualize_path.py
# Visualizes the solution path, obstacles, and uncertainty ellipses for the CCRRT planner.
#
# Features:
# - Reads static obstacles from file (obstacles.txt)
# - Reads planned path (solution_path.txt), RRT tree (rrt_tree.txt), and covariances (covariances.txt)
# - Animates the robot's path, showing uncertainty as an ellipse at each step
# - Supports saving the animation as a video (MP4) or displaying interactively
#
# File formats:
# - Obstacles: x, y, w, h[, dynamic]
# - Path: x, y, theta, v (theta/v may be unused)
# - Covariances: 2x2 matrix per line (flattened)
# - RRT tree: x1 y1 t1 x2 y2 t2 per edge
#
# This script helps visualize how the CCRRT algorithm handles static obstacles
# and propagates uncertainty (chance constraints) along the path.
# =======================================================================================

import os
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation, FFMpegWriter

def read_obstacles(file_path):
    static_obstacles = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 4:
                continue
            try:
                # Rectangle: x, y, w, h
                if len(parts) == 4:
                    x = float(parts[0])
                    y = float(parts[1])
                    w = float(parts[2])
                    h = float(parts[3])
                    static_obstacles.append(('rect', x, y, w, h))
                # Static circle: x, y, r, "circle"
                elif len(parts) == 4 and parts[3].strip() == "circle":
                    x = float(parts[0])
                    y = float(parts[1])
                    r = float(parts[2])
                    static_obstacles.append(('circle', x, y, r))
            except ValueError:
                continue
    return static_obstacles, []

def read_path(file_path):
    path = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            x, y, t, v = map(float, parts[:4])
            theta = 0.0  # Always set orientation to 0
            path.append((x, y, theta, v))
    return path

def read_covariances(file_path):
    covariances = []
    with open(file_path, 'r') as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 4:
                covariances.append(((vals[0], vals[1]), (vals[2], vals[3])))
    return covariances

def read_tree(file_path):
    segs = []
    with open(file_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 4:
                x1, y1, x2, y2 = map(float, parts)
            elif len(parts) == 6:
                x1, y1, _, x2, y2, _ = map(float, parts)
            else:
                continue
            segs.append(((x1, y1), (x2, y2)))
    return segs

def animate_car_path(obstacles, path, covariances, save_video=False):
    fig, ax = plt.subplots()
    
    # Only static obstacles
    static_obstacles, _ = obstacles
    print(f"[DEBUG] Found {len(static_obstacles)} static obstacles")

    # Draw static obstacles
    static_patches = []
    for obst in static_obstacles:
        if obst[0] == 'circle':
            _, x, y, r = obst
            circ = patches.Circle((x, y), r, linewidth=1, edgecolor='black', facecolor='gray', alpha=0.7)
            static_patches.append(circ)
            ax.add_patch(circ)
        elif obst[0] == 'rect':
            _, x, y, w, h = obst
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='black', facecolor='gray', alpha=0.7)
            static_patches.append(rect)
            ax.add_patch(rect)

    # Draw RRT tree
    for (x1, y1), (x2, y2) in read_tree("rrt_tree.txt"):
        ax.plot([x1, x2], [y1, y2], color='lightgray', linewidth=0.5, zorder=0)

    # Plot full path
    xs = [state[0] for state in path]
    ys = [state[1] for state in path]
    ax.plot(xs, ys, color='blue', label='Path')

    # Mark the actual goal (from obstacles or user input)
    goal_x, goal_y = 0.0, 0.0
    try:
        with open("obstacles.txt", "r") as f:
            for line in f:
                pass
    except Exception:
        pass

    # Mark the actual goal
    ax.plot(goal_x, goal_y, marker='*', color='gold', markersize=14, label='Goal')

    # Mark the closest point reached (last point in path)
    if path:
        ax.plot(xs[-1], ys[-1], marker='o', color='red', markersize=10, label='Closest Point Reached')

    # Mark the start
    if path:
        ax.plot(xs[0], ys[0], marker='s', color='green', markersize=10, label='Start')

    ax.set_aspect('equal', 'box')

    # Compute global min/max for all path and obstacles
    all_x = xs[:]
    all_y = ys[:]
    all_w = []
    all_h = []

    for obst in static_obstacles:
        if obst[0] == 'circle':
            _, x, y, r = obst
            all_x.append(x)
            all_y.append(y)
            all_w.append(r)
            all_h.append(r)
        elif obst[0] == 'rect':
            _, x, y, w, h = obst
            all_x.append(x)
            all_y.append(y)
            all_w.append(w)
            all_h.append(h)

    # --- Set fixed bounds to [-10, 10] for both axes ---
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Animated Car Path')
    ax.legend(loc='upper right')

    # Animation elements
    path_line, = ax.plot([], [], color='red', linewidth=2)

    car_length = 0.5
    car_width = 0.3
    car_patch = patches.Rectangle((0, 0), car_length, car_width, angle=0.0, color='red', zorder=5)
    ax.add_patch(car_patch)

    cov_ellipse = Ellipse((0, 0), 0, 0, edgecolor='orange', facecolor='none', lw=0.8, alpha=0.6)
    ax.add_patch(cov_ellipse)

    def init():
        path_line.set_data([], [])
        return (path_line, car_patch, cov_ellipse, *static_patches)

    def update(frame):
        x, y, theta, _ = path[frame]
        path_line.set_data(xs[:frame + 1], ys[:frame + 1])

        # Update car position
        dx = -car_length / 2
        dy = -car_width / 2
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
        offset = rot @ np.array([dx, dy])
        car_patch.set_xy((x + offset[0], y + offset[1]))
        car_patch.angle = np.degrees(theta)

        # Update covariance ellipse
        cov = np.array(covariances[frame])
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * 1.96 * np.sqrt(vals)
        cov_ellipse.set_center((x, y))
        cov_ellipse.width = width
        cov_ellipse.height = height
        cov_ellipse.angle = angle

        return (path_line, car_patch, cov_ellipse, *static_patches)

    # --- Save a static image before animation ---
    # Draw the full path and the last covariance ellipse for reference
    path_line.set_data(xs, ys)
    if path:
        x, y, theta, _ = path[-1]
        dx = -car_length / 2
        dy = -car_width / 2
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
        offset = rot @ np.array([dx, dy])
        car_patch.set_xy((x + offset[0], y + offset[1]))
        car_patch.angle = np.degrees(theta)
        cov = np.array(covariances[-1])
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * 1.96 * np.sqrt(vals)
        cov_ellipse.set_center((x, y))
        cov_ellipse.width = width
        cov_ellipse.height = height
        cov_ellipse.angle = angle

    plt.savefig("path_generated.png", dpi=200)
    print("[INFO] Saved static path image as path_generated.png")
    # Reset for animation
    path_line.set_data([], [])
    car_patch.set_xy((0, 0))
    car_patch.angle = 0.0
    cov_ellipse.set_center((0, 0))
    cov_ellipse.width = 0
    cov_ellipse.height = 0
    cov_ellipse.angle = 0

    ani = FuncAnimation(fig, update, frames=len(path),
                        init_func=init, blit=True, interval=50, repeat=False)

    if save_video:
        print("[INFO] Saving animation to car_path_animation.mp4 ...")
        ani.save("car_path_animation.mp4", writer=FFMpegWriter(fps=20))
        print("[DONE] MP4 animation saved.")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Animate and save car path with uncertainty.")
    parser.add_argument("--pathfile", type=str, required=True, help="Path file with x y theta v per line")
    parser.add_argument("--obstacles", type=str, required=True, help="Obstacle file in x,y,w,h format per line")
    parser.add_argument("--savevideo", action="store_true", help="Save animation as MP4 instead of showing it")
    args = parser.parse_args()

    path = read_path(args.pathfile)
    obstacles = read_obstacles(args.obstacles)

    cov_file = "covariances.txt"
    covariances = []
    if os.path.exists(cov_file):
        covariances = read_covariances(cov_file)
        print(f"[INFO] Loaded {len(covariances)} covariance entries from '{cov_file}'")
    else:
        print(f"[INFO] No covariance file found ('{cov_file}'). Ellipses will be zero-sized.")
    
    if len(covariances) < len(path):
        missing = len(path) - len(covariances)
        print(f"[WARN] Only {len(covariances)} covariances for {len(path)} path points; padding {missing} zeros.")
        zero_cov = ((0.0, 0.0), (0.0, 0.0))
        covariances.extend([zero_cov] * missing)

    animate_car_path(obstacles, path, covariances, save_video=args.savevideo)

if __name__ == "__main__":
    main()
