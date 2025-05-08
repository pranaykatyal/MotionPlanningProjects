import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
    return static_obstacles

def visualize_obstacles(obstacles):
    fig, ax = plt.subplots()
    for obst in obstacles:
        if obst[0] == 'circle':
            _, x, y, r = obst
            circ = patches.Circle((x, y), r, linewidth=1, edgecolor='black', facecolor='gray', alpha=0.7)
            ax.add_patch(circ)
        elif obst[0] == 'rect':
            _, x, y, w, h = obst
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='black', facecolor='gray', alpha=0.7)
            ax.add_patch(rect)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Obstacles Only Visualization')
    plt.savefig("obstacles_only.png", dpi=200)
    print("[INFO] Saved static obstacles image as obstacles_only.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize workspace obstacles only.")
    parser.add_argument("--obstacles", type=str, required=True, help="Obstacle file in x,y,w,h format per line")
    args = parser.parse_args()
    obstacles = read_obstacles(args.obstacles)
    visualize_obstacles(obstacles)

if __name__ == "__main__":
    main()
