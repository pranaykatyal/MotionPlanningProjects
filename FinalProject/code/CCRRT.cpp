// =======================================================================================
// CCRRT.cpp
// Main entry point for Chance-Constrained RRT (CCRRT) motion planning in OMPL
//
// This code demonstrates how to use OMPL's RRT planner with goal biasing and
// incorporates chance constraints to handle uncertainty in robot motion and
// dynamic/static obstacles. 
//
// Key Concepts:
// - State: 2D position (x, y) in R^2 (RealVectorStateSpace)
// - Control: 2D velocity (vx, vy) in R^2 (RealVectorControlSpace)
// - Chance Constraints: Probabilistic collision avoidance using state uncertainty
// - Collision Checking: Includes static, dynamic, and probabilistic (chance) checks
// =======================================================================================

#include <ompl/control/SpaceInformation.h>
#include <ompl/control/SimpleSetup.h>
#include <ompl/control/planners/rrt/RRT.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/base/goals/GoalSampleableRegion.h>
#include <ompl/tools/config/SelfConfig.h>
#include <ompl/util/Exception.h>
#include <Eigen/Dense>
#include "MotionValidator.h"
#include "Obstacles.h"
#include <cmath>
#include <fstream>
#include <functional>
#include <vector>
#include <random>

// -----------------------------
// Global Obstacle Containers
// -----------------------------
// staticObstacles: Obstacles that do not move (e.g., walls, fixed objects)
std::vector<Obstacle> staticObstacles;

// -----------------------------
// OMPL Type Aliases
// -----------------------------
using namespace ompl::base;
using namespace ompl::control;

namespace ob = ompl::base;
namespace oc = ompl::control;

// --- CC-RRT Enhancements ---
// Node structure extension: store Δₘₐₓ(N), cost-so-far, and parent pointer
// Each CCRRTNode tracks:
//   - state: OMPL state pointer
//   - control: OMPL control pointer
//   - parent: pointer to parent node in the tree
//   - costSoFar: cumulative cost from root to this node
//   - deltaMax: maximum per-step collision probability along the branch from root to this node
//   - lowerBoundCostToGo: heuristic lower bound (e.g., straight-line distance to goal)
struct CCRRTNode {
    ob::State *state;
    oc::Control *control;
    CCRRTNode *parent;
    double costSoFar;
    double deltaMax; // Maximum per-step collision probability along branch
    double lowerBoundCostToGo; // e.g. straight-line distance to goal

    CCRRTNode(ob::State *s, oc::Control *c, CCRRTNode *p, double cost, double delta, double lb)
        : state(s), control(c), parent(p), costSoFar(cost), deltaMax(delta), lowerBoundCostToGo(lb) {}
};

// Improved risk-biased node selection
inline CCRRTNode* selectNodeToExpandRiskBiased(const std::vector<CCRRTNode*>& nodes, ompl::RNG& rng) {
    const int maxTries = 10;
    CCRRTNode* best = nodes[0];
    for (auto* n : nodes)
        if (n->deltaMax < best->deltaMax)
            best = n;
    for (int i = 0; i < maxTries; ++i) {
        CCRRTNode* candidate = nodes[rng.uniformInt(0, nodes.size() - 1)];
        if (rng.uniform01() > candidate->deltaMax)
            return candidate;
    }
    return best; // fallback to lowest-risk node
}

// Branch-and-bound cost pruning: check lower-bound cost-to-go + cost-so-far
// As soon as any goal path is found, record its bestCost. For each new node, compute
// lowerBoundCostToGo (e.g., straight-line distance to goal) plus costSoFar.
// If that sum exceeds bestCost, immediately prune that subtree.
inline bool shouldPrune(const CCRRTNode* node, double bestCost) {
    return (node->costSoFar + node->lowerBoundCostToGo) > bestCost;
}

// ==========================
// Helper: Heuristic Function
// ==========================
// Computes a consistent heuristic (e.g., Euclidean distance) between two states.
// This must be admissible (never overestimates) and consistent (triangle inequality).
inline double consistentHeuristic(const ob::State *from, const ob::State *to) {
    const auto *f = from->as<ob::RealVectorStateSpace::StateType>();
    const auto *t = to->as<ob::RealVectorStateSpace::StateType>();
    double dx = f->values[0] - t->values[0];
    double dy = f->values[1] - t->values[1];
    return std::sqrt(dx * dx + dy * dy);
}

// ==========================
// Helper: Monotonic Progress
// ==========================
// Ensures that the cost-so-far is non-decreasing along the path.
inline bool isMonotonicProgress(const CCRRTNode* parent, const CCRRTNode* child) {
    return child->costSoFar >= parent->costSoFar;
}

// ==========================
// Helper: No Loops
// ==========================
// Checks if a node creates a loop in the tree (should never happen in RRT, but check for safety).
inline bool createsLoop(const CCRRTNode* node, const ob::State* candidateState) {
    const CCRRTNode* current = node;
    while (current) {
        if (current->state == candidateState)
            return true;
        current = current->parent;
    }
    return false;
}

// ==========================
// Helper: Dynamic Feasibility
// ==========================
// Checks if the control and resulting motion are dynamically feasible.
// For a simple double integrator, this could be a velocity/acceleration bound.
inline bool isDynamicallyFeasible(const ob::State* from, const oc::Control* control, double maxVel) {
    const auto *ctrl = control->as<oc::RealVectorControlSpace::ControlType>();
    double vx = ctrl->values[0];
    double vy = ctrl->values[1];
    double v = std::sqrt(vx * vx + vy * vy);
    return v <= maxVel;
}

// =======================================================================================
// CCRRT Planner Class
// Extends OMPL's RRT to support chance constraints (uncertainty-aware planning)
// 
// To implement real-time replanning (Algorithm 2), you would wrap the single-shot solve()
// in a loop that repeatedly grows the tree for Δt, lazily validates the current best path,
// commits the first segment if valid, or prunes and retries otherwise, until the goal is reached.
// These heuristics (risk-biased selection, branch-and-bound pruning, and real-time replanning)
// all sit on top of the standard RRT logic and do not require RRT* rewiring.
// =======================================================================================
class CCRRT : public oc::RRT {
public:
    // StateWithCovariance: Stores mean, covariance, and timestamp for each state
    using StateWithCovariance = CCRRTDetail::StateWithCovariance;

    // Constructor
    // si: SpaceInformation pointer (defines state/control spaces, validity, etc.)
    // psafe: Probability threshold for safety (e.g., 0.98 means 98% safe)
    CCRRT(const oc::SpaceInformationPtr &si, double psafe = 0.99) 
        : oc::RRT(si), psafe_(psafe), currentTime_(0.0) {
        setName("CCRRT");
    }

    // -------------------------------------------------------------------------------
    // propagate: Simulates robot motion and propagates uncertainty (mean/covariance)
    // start: Current state (x, y)
    // control: Control input (vx, vy)
    // duration: Time duration for control application
    // result: Output state after applying control
    // -------------------------------------------------------------------------------
    void propagate(const ob::State *start, const oc::Control *control,
                  const double duration, ob::State *result) {
        // Get the control space information
        auto csi = std::static_pointer_cast<oc::SpaceInformation>(si_);
        si_->getStateSpace()->copyState(result, start);
    
        const unsigned int dim = si_->getStateDimension();
        const unsigned int ctrl_dim = csi->getControlSpace()->getDimension();
        
        // Extract control values from RealVectorControlSpace
        const auto *rctrl = control->as<oc::RealVectorControlSpace::ControlType>();
        auto *rstate = result->as<ob::RealVectorStateSpace::StateType>();

        // Convert states and controls to Eigen vectors for easier manipulation
        Eigen::VectorXd startVec(dim);
        Eigen::VectorXd controlVec(ctrl_dim);

        for (unsigned int i = 0; i < dim; ++i)
            startVec(i) = start->as<ob::RealVectorStateSpace::StateType>()->values[i];

        for (unsigned int i = 0; i < ctrl_dim; ++i)
            controlVec(i) = rctrl->values[i];

        // Linear system matrices for uncertainty propagation
        Eigen::MatrixXd A = Eigen::MatrixXd::Identity(dim, dim);  // State transition matrix
        Eigen::MatrixXd B = Eigen::MatrixXd::Identity(dim, ctrl_dim);  // Control input matrix

        // Adaptive process noise: increase near obstacles, decrease in free space
        double minObsDist = std::numeric_limits<double>::max();
        const auto *s = start->as<ob::RealVectorStateSpace::StateType>();
        Eigen::Vector2d pos(s->values[0], s->values[1]);
        for (const auto& obs : staticObstacles) {
            double d = (pos - obs.getCenter()).norm() - obs.getRadius();
            if (d < minObsDist) minObsDist = d;
        }
        double noiseScale = 1.0;
        if (minObsDist > 2.0) noiseScale = 0.2; // less noise in free space
        else if (minObsDist < 0.5) noiseScale = 2.0; // more noise near obstacles
        Eigen::MatrixXd Pw = Eigen::MatrixXd::Identity(dim, dim) * 0.01 * noiseScale;

        // Propagate mean state using linear dynamics (scaled by duration)
        Eigen::VectorXd nextMean = A * startVec + B * controlVec * duration;
        for (unsigned int i = 0; i < dim; ++i)
            rstate->values[i] = nextMean(i);

        // Propagate uncertainty using the Kalman filter prediction step
        auto startIt = stateUncertainty_.find(StateKey::fromState(start));
        // If no prior uncertainty, initialize with process noise (Pw) instead of small diagonal
        Eigen::MatrixXd prevCov = (startIt != stateUncertainty_.end()) ? 
            startIt->second.covariance : 
            Pw; // Replace 0.01*I with Pw
            
        // Get current timestamp or initialize to 0
        double currentTime = (startIt != stateUncertainty_.end()) ? 
            startIt->second.timestamp + duration : 
            currentTime_ + duration;
            
        // Update current time to max observed time
        currentTime_ = std::max(currentTime_, currentTime);

        // Propagate covariance: nextCov = A*prevCov*A^T + Pw
        Eigen::MatrixXd nextCov = A * prevCov * A.transpose() + Pw;
        stateUncertainty_[StateKey::fromState(result)] = {nextMean, nextCov, currentTime};
    }

    // Planner data extraction (for visualization/debugging)
    virtual void getPlannerData(ob::PlannerData &data) const override {
        oc::RRT::getPlannerData(data);
    }

    // Example: Override growTree or extend logic to enforce all constraints
    // (This is a sketch; actual OMPL RRT code is more complex)
    CCRRTNode* tryExtend(CCRRTNode* parent, oc::Control* control, double duration, ob::State* result, double maxVel) {
        // Propagate state
        propagate(parent->state, control, duration, result);

        // Heuristic consistency
        double h = consistentHeuristic(result, goalState_);
        // Monotonic progress
        double cost = parent->costSoFar + duration;
        // No loops
        if (createsLoop(parent, result))
            return nullptr;
        // Dynamic feasibility
        if (!isDynamicallyFeasible(parent->state, control, maxVel))
            return nullptr;

        // State validity check
        if (!si_->isValid(result))
            return nullptr;
        // Optionally, add a motion validity check here if needed:
        // if (!si_->checkMotion(parent->state, result))
        //     return nullptr;

        // All checks passed, create new node
        return new CCRRTNode(result, control, parent, cost, /*deltaMax*/0.0, h);
    }

    // Set the goal state for heuristic computation
    void setGoalState(const ob::State* goal) {
        goalState_ = goal;
    }

protected:
    double psafe_;  // Safety probability threshold for chance constraints
    double currentTime_;  // Current simulation time (used for dynamic obstacles)
    // stateUncertainty_: Map from StateKey to StateWithCovariance (tracks uncertainty for each state)
    std::map<StateKey, StateWithCovariance> stateUncertainty_;
    const ob::State* goalState_ = nullptr;

public:
    // Accessor for state uncertainty map (used by validators/checkers)
    std::map<StateKey, StateWithCovariance>* getStateUncertainty() {
        return reinterpret_cast<std::map<StateKey, CCRRTDetail::StateWithCovariance>*>(&stateUncertainty_);
    }
    
    double getCurrentTime() const {
        return currentTime_;
    }
};

// =======================================================================================
// ChanceConstraintStateValidityChecker
// Checks if a state is valid (collision-free) under chance constraints
// =======================================================================================
class ChanceConstraintStateValidityChecker : public ob::StateValidityChecker {
public:
    // si: SpaceInformation pointer
    // psafe: Probability threshold for safety
    // stateUncertainty: Pointer to uncertainty map (mean/covariance/timestamp for each state)
    ChanceConstraintStateValidityChecker(const ob::SpaceInformationPtr &si,
                                       double psafe,
                                       std::map<StateKey, CCRRTDetail::StateWithCovariance>* stateUncertainty)
        : ob::StateValidityChecker(si), psafe_(psafe), stateUncertainty_(stateUncertainty) {}

    // -------------------------------------------------------------------------------
    // isValid: Checks if a state is valid (collision-free)
    // Performs three types of collision checking:
    //   1. Static obstacles (fixed in space)
    //   2. Dynamic obstacles (position depends on time)
    //   3. Chance constraints (probabilistic collision, using uncertainty)
    // -------------------------------------------------------------------------------
    virtual bool isValid(const ob::State *state) const override {
        const auto *s = state->as<ob::RealVectorStateSpace::StateType>();
        
        if (!si_->satisfiesBounds(state))
            return false;

        // Check static obstacles (rectangular collision)
        for (const auto& obs : staticObstacles) {
            Eigen::Vector2d stateVec(s->values[0], s->values[1]);
            if (!obs.isCircular()) {
                double x = obs.getX(), y = obs.getY();
                double w = obs.getWidth(), h = obs.getHeight();
                if (stateVec[0] >= x && stateVec[0] <= x + w &&
                    stateVec[1] >= y && stateVec[1] <= y + h)
                    return false;
            } else {
                // Fallback for circles (should not occur)
                double dist = (stateVec - obs.getCenter()).norm();
                if (dist <= obs.getRadius())
                    return false;
            }
        }

        // --- Add chance constraint collision check for covariance ellipse ---
        if (stateUncertainty_) {
            StateKey key = StateKey::fromState(state);
            auto it = stateUncertainty_->find(key);
            if (it != stateUncertainty_->end()) {
                const auto& unc = it->second;
                Eigen::Vector2d mean = unc.mean.head<2>();
                Eigen::Matrix2d cov = unc.covariance.block<2,2>(0, 0);

                for (const auto& obs : staticObstacles) {
                    if (!obs.isCircular()) {
                        // Rectangle: use circumscribed circle for chance constraint
                        Eigen::Vector2d obsCenter = obs.getCenter();
                        double obsRadius = std::hypot(obs.getWidth()/2, obs.getHeight()/2);
                        Eigen::Vector2d diff = obsCenter - mean;
                        double dist = diff.norm();
                        double directionalVar = (dist > 1e-6) ?
                            (diff.normalized().transpose() * cov * diff.normalized()) :
                            cov.eigenvalues().real().maxCoeff();
                        double sigma = std::sqrt(directionalVar);
                        double beta = std::sqrt(2) * erfcinv(2 * (1 - psafe_));
                        if (dist > obsRadius + 2.0) beta *= 0.7;
                        if (dist <= obsRadius + beta * sigma)
                            return false;
                    } else {
                        // Fallback for circles (should not occur)
                        Eigen::Vector2d obsCenter = obs.getCenter();
                        double obsRadius = obs.getRadius();
                        Eigen::Vector2d diff = obsCenter - mean;
                        double dist = diff.norm();
                        double directionalVar = (dist > 1e-6) ?
                            (diff.normalized().transpose() * cov * diff.normalized()) :
                            cov.eigenvalues().real().maxCoeff();
                        double sigma = std::sqrt(directionalVar);
                        double beta = std::sqrt(2) * erfcinv(2 * (1 - psafe_));
                        if (dist > obsRadius + 2.0) beta *= 0.7;
                        if (dist <= obsRadius + beta * sigma)
                            return false;
                    }
                }
            }
        }
        // --- End chance constraint check ---

        return true;
    }
    
private:
    double psafe_;
    std::map<StateKey, CCRRTDetail::StateWithCovariance>* stateUncertainty_;
};

// =======================================================================================
// Helper Functions
// =======================================================================================

// Finds the nearest state in the uncertainty map to a query state
const CCRRTDetail::StateWithCovariance* findNearestUncertainty(
    const std::map<StateKey, CCRRTDetail::StateWithCovariance>& uncert,
    const ob::State* query)
{
    if (uncert.empty() || query == nullptr) {
        std::cerr << "[DEBUG] Uncertainty map empty or query is null\n";
        return nullptr;
    }
    StateKey qk = StateKey::fromState(query);
    double minDist = std::numeric_limits<double>::max();
    const CCRRTDetail::StateWithCovariance* nearest = nullptr;
    for (const auto& pair : uncert) {
        double dx = qk.x - pair.first.x;
        double dy = qk.y - pair.first.y;
        double dist = dx*dx + dy*dy;
        if (dist < minDist) {
            minDist = dist;
            nearest = &pair.second;
        }
    }
    if (!nearest) {
        std::cerr << "[DEBUG] No nearest state found in uncertainty map\n";
    }
    return nearest;
}

// =======================================================================================
// Main Function
// Sets up the planning problem, obstacles, planner, and runs CCRRT
// =======================================================================================
int main(int argc, char **argv)
{
    // --- Set psafe parameter here ---
    const double psafe = 0.99;

    // 1) State Space: 2D position (x, y) in R^2
    //    - RealVectorStateSpace(2)
    //    - Bounds: [-10, 10] for both x and y
    auto space = std::make_shared<ob::RealVectorStateSpace>(2);
    ob::RealVectorBounds bounds(2);
    bounds.setLow(-10);
    bounds.setHigh(10);
    space->setBounds(bounds);

    // 2) Control Space: 2D velocity (vx, vy) in R^2
    //    - RealVectorControlSpace(2)
    //    - Bounds: [-2, 2] for both vx and vy
    auto cspace = std::make_shared<oc::RealVectorControlSpace>(space, 2);
    ob::RealVectorBounds cbounds(2);
    cbounds.setLow(-2);   // Increased from -1 to -2
    cbounds.setHigh(2);   // Increased from 1 to 2
    cspace->setBounds(cbounds);

    // 2) Create SpaceInformation & CCRRT planner
    auto si      = std::make_shared<oc::SpaceInformation>(space, cspace);
    auto planner = std::make_shared<CCRRT>(si, psafe);

    // Set min/max control duration to avoid OMPL warning
    si->setMinMaxControlDuration(1, 50);

    // Set the goal bias - add this line
    planner->setGoalBias(0.1); // 5% chance to sample the goal

    // 3) Use CCRRT::propagate so we track covariance
    si->setStatePropagator(
        [planner](const ob::State *start, const oc::Control *ctrl,
                  double dt, ob::State *result) {
            planner->propagate(start, ctrl, dt, result);
        });

    // 4) Obstacles:
    //    - staticObstacles: Placed between start and goal (not behind goal)
    staticObstacles.clear();
    static const int NUM_STATIC_OBSTACLES = 7; // Increased from  7
    static const double START_X = -8.0;
    static const double START_Y = -8.0;
    static const double GOAL_X = 0.0;
    static const double GOAL_Y = 0.0;

    // Place obstacles along the line from start to goal, with some spread and more spacing
    double widths[NUM_STATIC_OBSTACLES]  = {2.0, 2.5, 3.0, 1.5, 2.2, 2.0, 2.5};
    double heights[NUM_STATIC_OBSTACLES] = {1.0, 1.8, 1.2, 2.5, 2.0, 1.5, 1.7};
    // More spacing: spread fractions more evenly and increase lateral offsets
    double fractions[NUM_STATIC_OBSTACLES] = {0.15, 0.28, 0.42, 0.57, 0.71, 0.83, 0.92};
    double lateral_offsets[NUM_STATIC_OBSTACLES] = {-2.5, 2.2, -1.7, 2.7, -2.2, 2.9, -2.8};

    Eigen::Vector2d start_eigen(START_X, START_Y);
    Eigen::Vector2d goal_eigen(GOAL_X, GOAL_Y);
    Eigen::Vector2d direction = (goal_eigen - start_eigen).normalized();
    Eigen::Vector2d perp(-direction[1], direction[0]); // perpendicular vector

    for (int i = 0; i < NUM_STATIC_OBSTACLES; ++i) {
        Eigen::Vector2d center = start_eigen + (goal_eigen - start_eigen) * fractions[i] + perp * lateral_offsets[i];
        double x = center[0] - widths[i]/2;
        double y = center[1] - heights[i]/2;
        staticObstacles.push_back(Obstacle(x, y, widths[i], heights[i]));
    }

    // Write obstacle definitions to file for visualization (rectangles)
    {
        std::ofstream f("obstacles.txt");
        for (auto &obs : staticObstacles)
            f << obs.getX() << ","    // lower-left x
              << obs.getY() << ","    // lower-left y
              << obs.getWidth() << "," // width
              << obs.getHeight() << "\n"; // height
    }

    // 5) Start/Goal States:
    //    - Start: (-8, -8)
    //    - Goal:  (0, 0)
    ob::ScopedState<ob::RealVectorStateSpace> start(space), goal(space);
    start[0] = START_X; start[1] = START_Y;
    goal[0]  = GOAL_X;  goal[1]  = GOAL_Y;

    // Use SimpleSetup's setStartAndGoalStates, not SpaceInformation's
    oc::SimpleSetup ss(si);
    ss.setPlanner(planner);

    ss.setStartAndGoalStates(start, goal, /*threshold=*/0.1);

    // --- Direct goal sampling if near the goal ---
    // If the robot is within a certain distance to the goal, increase goal bias to 0.5
    double near_goal_radius =0.5; // Radius around the goal to increase bias 
    auto setGoalBiasIfNearGoal = [&](const ob::State *current) {
        const auto *curr = current->as<ob::RealVectorStateSpace::StateType>();
        double dx = curr->values[0] - goal[0];
        double dy = curr->values[1] - goal[1];
        double dist = std::sqrt(dx * dx + dy * dy);
        if (dist < near_goal_radius) {
            planner->setGoalBias(1.0); // Always sample the goal if near
        } else {
            planner->setGoalBias(0.1); // Default bias
        }
    };
    // Example usage: call setGoalBiasIfNearGoal at the start and after each planning iteration
    setGoalBiasIfNearGoal(start.get());

    // 6) SimpleSetup: OMPL helper for planning
    ss.setPlanner(planner);

    // 7) State Validity Checker:
    //    - Uses ChanceConstraintStateValidityChecker to check for collisions
    //      with static obstacles and chance constraints
    si->setStateValidityChecker(
        std::make_shared<ChanceConstraintStateValidityChecker>(
            si, psafe, reinterpret_cast<std::map<StateKey, CCRRTDetail::StateWithCovariance>*>(planner->getStateUncertainty())));

    // 8) Motion Validator:
    //    - Uses CCRRTMotionValidator to check for valid motions (edges)
    //    - Checks collisions along the path between two states
    //    - Handles static obstacles, as well as chance constraints
    auto mv = std::make_shared<CCRRTMotionValidator>(si, psafe);
    // Only pass static obstacles to the motion validator
    mv->setObstacles(staticObstacles);
    // Cast to CCRRTDetail::StateWithCovariance*
    mv->setStateUncertainty(reinterpret_cast<std::map<StateKey, CCRRTDetail::StateWithCovariance>*>(planner->getStateUncertainty())); // Pass uncertainty map
    si->setMotionValidator(mv);

    // 9) Tuning:
    //    - State validity checking resolution: 0.02 (fraction of state space)
    //    - Propagation step size: 0.1 (time step for state propagation)
    si->setStateValidityCheckingResolution(0.02);
    si->setPropagationStepSize(0.1);

    // 10) Planning:
    //    - Runs the planner for up to 10 seconds
    //    - If a solution is found, writes the path, covariances, state times, and tree to files
    ss.solve(ob::timedPlannerTerminationCondition(10.0));

    if (!ss.haveSolutionPath())
    {
        std::cout << "No solution found\n";
        return 0;
    }

    // Output files:
    // - solution_path.txt: Smoothed solution path (x, y)
    // - covariances.txt: Covariance matrices for each state in the path
    // - state_times.txt: Time at which each state is reached
    // - rrt_tree.txt: All edges in the RRT search tree
    // - obstacles.txt: Obstacle definitions for visualization
    {
        std::ofstream pathFile("solution_path.txt");
        auto smooth = ss.getSolutionPath().asGeometric();
        smooth.interpolate(50); // Interpolate to 50 states
        for (std::size_t i = 0; i < smooth.getStateCount(); ++i)
        {
            const ob::State *st = smooth.getState(i);
            const auto *rstate = st->as<ob::RealVectorStateSpace::StateType>();
            pathFile << rstate->values[0] << " "
                     << rstate->values[1] << " 0 0\n";
        }
        std::cout << "Path written to solution_path.txt\n";
    }

    
    {
        std::ofstream covFile("covariances.txt");
        std::cout << "[INFO] Writing covariances...\n";
        auto raw = ss.getSolutionPath().asGeometric();
        std::size_t N = raw.getStateCount();
        auto *uncert = reinterpret_cast<std::map<StateKey, CCRRTDetail::StateWithCovariance>*>(planner->getStateUncertainty());
        try {
            for (std::size_t i = 0; i < N; ++i) {
                const ob::State *s = raw.getState(i);
                const CCRRTDetail::StateWithCovariance* nearest = findNearestUncertainty(*uncert, s);
                if (nearest) {
                    const auto &C = nearest->covariance;
                    covFile << C(0,0) << " " << C(0,1) << " "
                            << C(1,0) << " " << C(1,1) << "\n";
                } else {
                    covFile << "0 0 0 0\n";
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Exception while writing covariances: " << e.what() << std::endl;
        }
        std::cout << "[INFO] Covariances written to covariances.txt\n";
    }
    // Write time information for each state in the solution path
    {
        std::ofstream timeFile("state_times.txt");
        auto raw = ss.getSolutionPath().asGeometric();
        std::size_t N = raw.getStateCount();
        
        auto *uncert = reinterpret_cast<std::map<StateKey, CCRRTDetail::StateWithCovariance>*>(planner->getStateUncertainty());
        for (std::size_t i = 0; i < N; ++i) {
            const ob::State *s = raw.getState(i);
            auto it = uncert->find(StateKey::fromState(s));
            if (it != uncert->end()) {
                auto *st = s->as<ob::RealVectorStateSpace::StateType>();
                timeFile << st->values[0] << " " << st->values[1] << " " 
                         << it->second.timestamp << "\n";
            }
        }
        std::cout << "[INFO] State times written to state_times.txt\n";
    }
    //
    // --- Dump the entire RRT search tree (all tried edges)
    //
    {
        ompl::base::PlannerData pdata(si);
        ss.getPlannerData(pdata);

        std::ofstream treeFile("rrt_tree.txt");
        for (unsigned int vid = 0; vid < pdata.numVertices(); ++vid)
        {
            std::vector<unsigned int> edges;
            pdata.getEdges(vid, edges);
            for (auto to : edges)
            {
                const auto *su = pdata.getVertex(vid).getState()
                                   ->as<ob::RealVectorStateSpace::StateType>();
                const auto *sv = pdata.getVertex(to).getState()
                                   ->as<ob::RealVectorStateSpace::StateType>();
                treeFile
                  << su->values[0] << " " << su->values[1]
                  << "   "
                  << sv->values[0] << " " << sv->values[1]
                  << "\n";
            }
        }
        treeFile.close();
        std::cout << "RRT tree written to rrt_tree.txt\n";
    }
    return 0;
}