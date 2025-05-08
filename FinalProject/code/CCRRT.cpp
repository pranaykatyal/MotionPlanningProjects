#include <ompl/control/SpaceInformation.h>
#include <ompl/control/SimpleSetup.h>
#include <ompl/control/planners/rrt/RRT.h>
#include <ompl/tools/benchmark/Benchmark.h>
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

// List of static obstacles in the workspace, used for collision and uncertainty calculations.
std::vector<Obstacle> staticObstacles;

using namespace ompl::base;
using namespace ompl::control;

namespace ob = ompl::base;
namespace oc = ompl::control;

struct CCRRTNode {
    ob::State *state;           // Pointer to OMPL state (position)
    oc::Control *control;       // Pointer to OMPL control (velocity)
    CCRRTNode *parent;          // Parent node in the tree
    double costSoFar;           // Accumulated cost from start to this node
    double deltaMax;            // Maximum per-step collision probability along the path to this node
    double lowerBoundCostToGo;  // Heuristic lower bound to goal (e.g., Euclidean distance)
    CCRRTNode(ob::State *s, oc::Control *c, CCRRTNode *p, double cost, double delta, double lb)
        : state(s), control(c), parent(p), costSoFar(cost), deltaMax(delta), lowerBoundCostToGo(lb) {}
};

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
    return best;
}

inline bool shouldPrune(const CCRRTNode* node, double bestCost) {
    return (node->costSoFar + node->lowerBoundCostToGo) > bestCost;
}

inline double consistentHeuristic(const ob::State *from, const ob::State *to) {
    const auto *f = from->as<ob::RealVectorStateSpace::StateType>();
    const auto *t = to->as<ob::RealVectorStateSpace::StateType>();
    double dx = f->values[0] - t->values[0];
    double dy = f->values[1] - t->values[1];
    return std::sqrt(dx * dx + dy * dy);
}

inline bool isMonotonicProgress(const CCRRTNode* parent, const CCRRTNode* child) {
    return child->costSoFar >= parent->costSoFar;
}

inline bool createsLoop(const CCRRTNode* node, const ob::State* candidateState) {
    const CCRRTNode* current = node;
    while (current) {
        if (current->state == candidateState)
            return true;
        current = current->parent;
    }
    return false;
}

inline bool isDynamicallyFeasible(const ob::State* from, const oc::Control* control, double maxVel) {
    const auto *ctrl = control->as<oc::RealVectorControlSpace::ControlType>();
    double vx = ctrl->values[0];
    double vy = ctrl->values[1];
    double v = std::sqrt(vx * vx + vy * vy);
    return v <= maxVel;
}

class CCRRT : public oc::RRT {
public:
    using StateWithCovariance = CCRRTDetail::StateWithCovariance;

    // Constructor: initializes planner with OMPL SpaceInformation and safety probability psafe.
    CCRRT(const oc::SpaceInformationPtr &si, double psafe = 0.7) 
        : oc::RRT(si), psafe_(psafe), currentTime_(0.0) {
        setName("CCRRT");
    }

    // Propagate state and uncertainty using linear Gaussian model.
    // start: initial state, control: control input, duration: time, result: output state.
    void propagate(const ob::State *start, const oc::Control *control,
                  const double duration, ob::State *result) {
        auto csi = std::static_pointer_cast<oc::SpaceInformation>(si_);
        si_->getStateSpace()->copyState(result, start);
        const unsigned int dim = si_->getStateDimension();      // State dimension (2 for x,y)
        const unsigned int ctrl_dim = csi->getControlSpace()->getDimension(); // Control dimension (2 for vx,vy)
        const auto *rctrl = control->as<oc::RealVectorControlSpace::ControlType>();
        auto *rstate = result->as<ob::RealVectorStateSpace::StateType>();
        Eigen::VectorXd startVec(dim);      // State mean vector
        Eigen::VectorXd controlVec(ctrl_dim); // Control vector
        for (unsigned int i = 0; i < dim; ++i)
            startVec(i) = start->as<ob::RealVectorStateSpace::StateType>()->values[i];
        for (unsigned int i = 0; i < ctrl_dim; ++i)
            controlVec(i) = rctrl->values[i];
        Eigen::MatrixXd A = Eigen::MatrixXd::Identity(dim, dim); // State transition matrix
        Eigen::MatrixXd B = Eigen::MatrixXd::Identity(dim, ctrl_dim); // Control matrix

        // Compute minimum distance to obstacles for adaptive noise scaling.
        double minObsDist = std::numeric_limits<double>::max();
        const auto *s = start->as<ob::RealVectorStateSpace::StateType>();
        Eigen::Vector2d pos(s->values[0], s->values[1]);
        for (const auto& obs : staticObstacles) {
            double d = (pos - obs.getCenter()).norm() - obs.getRadius();
            if (d < minObsDist) minObsDist = d;
        }
        // Adaptive process noise scaling: less noise in free space, more near obstacles.
        double noiseScale = 1.0;
        if (minObsDist > 2.0) noiseScale = 0.2;
        else if (minObsDist < 0.5) noiseScale = 2.0;
        Eigen::MatrixXd Pw = Eigen::MatrixXd::Identity(dim, dim) * 0.01 * noiseScale; // Process noise

        // Kalman prediction: propagate mean and covariance.
        Eigen::VectorXd nextMean = A * startVec + B * controlVec * duration;
        for (unsigned int i = 0; i < dim; ++i)
            rstate->values[i] = nextMean(i);

        // Retrieve previous covariance from stateUncertainty_ map, or use process noise if not found.
        auto startIt = stateUncertainty_.find(StateKey::fromState(start));
        Eigen::MatrixXd prevCov = (startIt != stateUncertainty_.end()) ? 
            startIt->second.covariance : 
            Pw;
        double currentTime = (startIt != stateUncertainty_.end()) ? 
            startIt->second.timestamp + duration : 
            currentTime_ + duration;
        currentTime_ = std::max(currentTime_, currentTime);
        Eigen::MatrixXd nextCov = A * prevCov * A.transpose() + Pw;

        // Store propagated mean, covariance, and timestamp for this state.
        stateUncertainty_[StateKey::fromState(result)] = {nextMean, nextCov, currentTime};
    }

    // Collect planner data for OMPL (delegates to base RRT).
    virtual void getPlannerData(ob::PlannerData &data) const override {
        oc::RRT::getPlannerData(data);
    }

    // Try to extend the tree from parent using control and duration.
    // Returns new node if valid, nullptr otherwise.
    CCRRTNode* tryExtend(CCRRTNode* parent, oc::Control* control, double duration, ob::State* result, double maxVel) {
        propagate(parent->state, control, duration, result);
        double h = consistentHeuristic(result, goalState_);
        double cost = parent->costSoFar + duration;
        if (createsLoop(parent, result))
            return nullptr;
        if (!isDynamicallyFeasible(parent->state, control, maxVel))
            return nullptr;
        if (!si_->isValid(result))
            return nullptr;
        return new CCRRTNode(result, control, parent, cost, 0.0, h);
    }

    // Set the goal state for heuristic calculations.
    void setGoalState(const ob::State* goal) {
        goalState_ = goal;
    }
protected:
    double psafe_; // Probability threshold for safety (chance constraint)
    double currentTime_; // Current time in the planner (for uncertainty timestamps)
    std::map<StateKey, StateWithCovariance> stateUncertainty_; // Map: state key -> mean/covariance/timestamp
    const ob::State* goalState_ = nullptr; // Pointer to goal state
public:
    // Return pointer to the state uncertainty map (for use by validity checker, etc.)
    std::map<StateKey, StateWithCovariance>* getStateUncertainty() {
        return reinterpret_cast<std::map<StateKey, CCRRTDetail::StateWithCovariance>*>(&stateUncertainty_);
    }
    // Get current time (for debugging or logging)
    double getCurrentTime() const {
        return currentTime_;
    }
};

// State validity checker for chance constraints.
// Checks both deterministic and probabilistic (Gaussian) collision.
class ChanceConstraintStateValidityChecker : public ob::StateValidityChecker {
public:
    // Constructor: takes OMPL SpaceInformation, safety probability, and pointer to state uncertainty map.
    ChanceConstraintStateValidityChecker(const ob::SpaceInformationPtr &si,
                                       double psafe,
                                       std::map<StateKey, CCRRTDetail::StateWithCovariance>* stateUncertainty)
        : ob::StateValidityChecker(si), psafe_(psafe), stateUncertainty_(stateUncertainty) {}

    // Returns true if state is valid (not in collision, satisfies chance constraint).
    virtual bool isValid(const ob::State *state) const override {
        const auto *s = state->as<ob::RealVectorStateSpace::StateType>();
        if (!si_->satisfiesBounds(state))
            return false;
        // Deterministic collision check with all obstacles.
        for (const auto& obs : staticObstacles) {
            Eigen::Vector2d stateVec(s->values[0], s->values[1]);
            if (!obs.isCircular()) {
                double x = obs.getX(), y = obs.getY();
                double w = obs.getWidth(), h = obs.getHeight();
                if (stateVec[0] >= x && stateVec[0] <= x + w &&
                    stateVec[1] >= y && stateVec[1] <= y + h)
                    return false;
            } else {
                double dist = (stateVec - obs.getCenter()).norm();
                if (dist <= obs.getRadius())
                    return false;
            }
        }
        // Probabilistic collision check using Gaussian uncertainty.
        if (stateUncertainty_) {
            StateKey key = StateKey::fromState(state);
            auto it = stateUncertainty_->find(key);
            if (it != stateUncertainty_->end()) {
                const auto& unc = it->second;
                Eigen::Vector2d mean = unc.mean.head<2>();
                Eigen::Matrix2d cov = unc.covariance.block<2,2>(0, 0);
                for (const auto& obs : staticObstacles) {
                    // For rectangles, approximate by circumscribed circle.
                    if (!obs.isCircular()) {
                        Eigen::Vector2d obsCenter = obs.getCenter();
                        double obsRadius = std::hypot(obs.getWidth()/2, obs.getHeight()/2);
                        Eigen::Vector2d diff = obsCenter - mean;
                        double dist = diff.norm();
                        // Directional variance along mean-to-obstacle vector.
                        double directionalVar = (dist > 1e-6) ?
                            (diff.normalized().transpose() * cov * diff.normalized()) :
                            cov.eigenvalues().real().maxCoeff();
                        double sigma = std::sqrt(directionalVar);
                        // Beta: quantile for chance constraint (function of psafe).
                        double beta = std::sqrt(2) * erfcinv(2 * (1 - psafe_));
                        // Slightly relax beta for far obstacles.
                        if (dist > obsRadius + 2.0) beta *= 0.7;
                        // Chance constraint: mean must be outside obstacle + beta*sigma.
                        if (dist <= obsRadius + beta * sigma)
                            return false;
                    } else {
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
        return true;
    }
private:
    double psafe_; // Probability threshold for safety
    std::map<StateKey, CCRRTDetail::StateWithCovariance>* stateUncertainty_; // Pointer to uncertainty map
};

// Find the nearest state in the uncertainty map to the query state.
// Used for logging/visualization of covariances along the path.
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

void benchmarkCCRRT(oc::SimpleSetupPtr ss)
{
    ss->setPlanner(nullptr);
    auto pdef = ss->getProblemDefinition();
    if (!pdef || pdef->getStartStateCount() == 0 || !pdef->getGoal())
    {
        std::cerr << "[ERROR] Start and/or goal states are not set for benchmarking. Aborting benchmark." << std::endl;
        return;
    }
    auto si = ss->getSpaceInformation();
    si->setStateValidityChecker(
        std::make_shared<ChanceConstraintStateValidityChecker>(
            si, 0.7, nullptr));
    auto mv = std::make_shared<CCRRTMotionValidator>(si, 0.7);
    mv->setObstacles(staticObstacles);
    mv->setStateUncertainty(nullptr);
    si->setMotionValidator(mv);
    ss->setPlanner(std::make_shared<oc::RRT>(si));
    ss->setup();
    ompl::tools::Benchmark b(*ss.get(), "CCRRT_vs_RRT");
    auto rrt = std::make_shared<oc::RRT>(ss->getSpaceInformation());
    rrt->setProblemDefinition(ss->getProblemDefinition());
    rrt->setName("RRT");
    rrt->setGoalBias(0.1);
    b.addPlanner(rrt);
    std::vector<double> psafe_values = {0.5, 0.7, 0.9, 0.95, 0.99};
    for (double psafe : psafe_values)
    {
        auto ccrrt = std::make_shared<CCRRT>(ss->getSpaceInformation(), psafe);
        ccrrt->setProblemDefinition(ss->getProblemDefinition());
        ccrrt->setName("CCRRT_psafe_" + std::to_string(psafe));
        ccrrt->setGoalBias(0.1);
        auto ccrrt_si = std::dynamic_pointer_cast<oc::SpaceInformation>(ccrrt->getSpaceInformation());
        ccrrt_si->setStatePropagator(
            [ccrrt](const ob::State *start, const oc::Control *ctrl, double dt, ob::State *result) {
                ccrrt->propagate(start, ctrl, dt, result);
            });
        ccrrt_si->setStateValidityChecker(
            std::make_shared<ChanceConstraintStateValidityChecker>(
                ccrrt_si, psafe, reinterpret_cast<std::map<StateKey, CCRRTDetail::StateWithCovariance>*>(ccrrt->getStateUncertainty())));
        auto ccrrt_mv = std::make_shared<CCRRTMotionValidator>(ccrrt_si, psafe);
        ccrrt_mv->setObstacles(staticObstacles);
        ccrrt_mv->setStateUncertainty(reinterpret_cast<std::map<StateKey, CCRRTDetail::StateWithCovariance>*>(ccrrt->getStateUncertainty()));
        ccrrt_si->setMotionValidator(ccrrt_mv);
        b.addPlanner(ccrrt);
    }
    ompl::tools::Benchmark::Request request(60.0, 2048.0, 10, 0.5);
    b.benchmark(request);
    b.saveResultsToFile("CCRRTbenchmark.log");
    std::cout << "Benchmark complete. Results saved to CCRRTbenchmark.log" << std::endl;
}

int main(int argc, char **argv)
{
    // ====== SETUP: Define state and control spaces, bounds, and obstacles ======
    const double psafe = 0.7;
    auto space = std::make_shared<ob::RealVectorStateSpace>(2);
    ob::RealVectorBounds bounds(2);
    bounds.setLow(-10);
    bounds.setHigh(10);
    space->setBounds(bounds);
    auto cspace = std::make_shared<oc::RealVectorControlSpace>(space, 2);
    ob::RealVectorBounds cbounds(2);
    cbounds.setLow(-2);
    cbounds.setHigh(2);
    cspace->setBounds(cbounds);
    auto si      = std::make_shared<oc::SpaceInformation>(space, cspace);
    auto planner = std::make_shared<CCRRT>(si, psafe);

    // ====== SETUP: Configure OMPL parameters and obstacles ======
    si->setMinMaxControlDuration(1, 10);
    si->setStateValidityChecker(
        std::make_shared<ChanceConstraintStateValidityChecker>(
            si, psafe, reinterpret_cast<std::map<StateKey, CCRRTDetail::StateWithCovariance>*>(planner->getStateUncertainty())));
    auto mv = std::make_shared<CCRRTMotionValidator>(si, psafe);
    mv->setObstacles(staticObstacles);
    mv->setStateUncertainty(reinterpret_cast<std::map<StateKey, CCRRTDetail::StateWithCovariance>*>(planner->getStateUncertainty()));
    si->setMotionValidator(mv);
    si->setStatePropagator(
        [planner](const ob::State *start, const oc::Control *ctrl,
                  double dt, ob::State *result) {
            planner->propagate(start, ctrl, dt, result);
        });

    // ====== OBSTACLE GENERATION ======
    staticObstacles.clear();
    static const int NUM_STATIC_OBSTACLES = 7;
    static const double START_X = -8.0;
    static const double START_Y = -8.0;
    static const double GOAL_X = 0.0;
    static const double GOAL_Y = 0.0;
    double widths[NUM_STATIC_OBSTACLES]  = {2.0, 2.5, 3.0, 1.5, 2.2, 2.0, 2.5};
    double heights[NUM_STATIC_OBSTACLES] = {1.0, 1.8, 1.2, 2.5, 2.0, 1.5, 1.7};
    double fractions[NUM_STATIC_OBSTACLES] = {0.15, 0.28, 0.42, 0.57, 0.71, 0.83, 0.92};
    double lateral_offsets[NUM_STATIC_OBSTACLES] = {-2.5, 2.2, -1.7, 2.7, -2.2, 2.9, -2.8};
    Eigen::Vector2d start_eigen(START_X, START_Y);
    Eigen::Vector2d goal_eigen(GOAL_X, GOAL_Y);
    Eigen::Vector2d direction = (goal_eigen - start_eigen).normalized();
    Eigen::Vector2d perp(-direction[1], direction[0]);
    for (int i = 0; i < NUM_STATIC_OBSTACLES; ++i) {
        Eigen::Vector2d center = start_eigen + (goal_eigen - start_eigen) * fractions[i] + perp * lateral_offsets[i];
        double x = center[0] - widths[i]/2;
        double y = center[1] - heights[i]/2;
        staticObstacles.push_back(Obstacle(x, y, widths[i], heights[i]));
    }
    {
        std::ofstream f("obstacles.txt");
        for (auto &obs : staticObstacles)
            f << obs.getX() << ","
              << obs.getY() << ","
              << obs.getWidth() << ","
              << obs.getHeight() << "\n";
    }

    // ====== START AND GOAL SETUP ======
    ob::ScopedState<ob::RealVectorStateSpace> start(space), goal(space);
    start[0] = START_X; start[1] = START_Y;
    goal[0]  = GOAL_X;  goal[1]  = GOAL_Y;
    oc::SimpleSetup ss(si);
    ss.setPlanner(planner);
    ss.setStartAndGoalStates(start, goal, 0.25);

    // ====== GOAL BIAS ADJUSTMENT NEAR GOAL ======
    double near_goal_radius = 0.5;
    auto setGoalBiasIfNearGoal = [&](const ob::State *current) {
        const auto *curr = current->as<ob::RealVectorStateSpace::StateType>();
        double dx = curr->values[0] - goal[0];
        double dy = curr->values[1] - goal[1];
        double dist = std::sqrt(dx * dx + dy * dy);
        if (dist < near_goal_radius) {
            planner->setGoalBias(1.0);
        } else {
            planner->setGoalBias(0.1);
        }
    };
    setGoalBiasIfNearGoal(start.get());

    // ====== OMPL PLANNER PARAMETERS ======
    si->setStateValidityCheckingResolution(0.02);
    si->setPropagationStepSize(0.1);

    // ====== SOLVE THE PLANNING PROBLEM (FIRST RUN) ======
    ss.solve(ob::timedPlannerTerminationCondition(10.0));
    if (!ss.haveSolutionPath())
    {
        std::cout << "No solution found\n";
        return 0;
    }

    // ====== OUTPUT: Write solution path, path length, covariances, state times, and tree ======
    {
        // Write the planned path to file
        std::ofstream pathFile("solution_path.txt");
        auto smooth = ss.getSolutionPath().asGeometric();
        smooth.interpolate(50);
        for (std::size_t i = 0; i < smooth.getStateCount(); ++i)
        {
            const ob::State *st = smooth.getState(i);
            const auto *rstate = st->as<ob::RealVectorStateSpace::StateType>();
            pathFile << rstate->values[0] << " "
                     << rstate->values[1] << " 0 0\n";
        }
        std::cout << "Path written to solution_path.txt\n";
        double pathLength = 0.0;
        for (std::size_t i = 1; i < smooth.getStateCount(); ++i) {
            const auto *s1 = smooth.getState(i-1)->as<ob::RealVectorStateSpace::StateType>();
            const auto *s2 = smooth.getState(i)->as<ob::RealVectorStateSpace::StateType>();
            double dx = s2->values[0] - s1->values[0];
            double dy = s2->values[1] - s1->values[1];
            pathLength += std::sqrt(dx*dx + dy*dy);
        }
        std::cout << "Total path length: " << pathLength << std::endl;
    }
    {
        // Write covariance matrices along the path
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
    {
        // Write state times (timestamps) along the path
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
    {
        // Write the RRT tree edges to file
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

    // ====== INTERACTIVE SECTION: Plan or Benchmark ======
    int choice;
    do
    {
        std::cout << "Plan or Benchmark? " << std::endl;
        std::cout << " (1) Plan" << std::endl;
        std::cout << " (2) Benchmark" << std::endl;
        std::cin >> choice;
    } while (choice < 1 || choice > 2);

    if (choice == 1)
    {
        // ====== RE-RUN PLANNER AND OUTPUT RESULTS ======
        ss.solve(ob::timedPlannerTerminationCondition(10.0));
        if (!ss.haveSolutionPath())
        {
            std::cout << "No solution found\n";
            return 0;
        }
        {
            // Write the planned path to file
            std::ofstream pathFile("solution_path.txt");
            auto smooth = ss.getSolutionPath().asGeometric();
            smooth.interpolate(50);
            for (std::size_t i = 0; i < smooth.getStateCount(); ++i)
            {
                const ob::State *st = smooth.getState(i);
                const auto *rstate = st->as<ob::RealVectorStateSpace::StateType>();
                pathFile << rstate->values[0] << " "
                         << rstate->values[1] << " 0 0\n";
            }
            std::cout << "Path written to solution_path.txt\n";
            double pathLength = 0.0;
            for (std::size_t i = 1; i < smooth.getStateCount(); ++i) {
                const auto *s1 = smooth.getState(i-1)->as<ob::RealVectorStateSpace::StateType>();
                const auto *s2 = smooth.getState(i)->as<ob::RealVectorStateSpace::StateType>();
                double dx = s2->values[0] - s1->values[0];
                double dy = s2->values[1] - s1->values[1];
                pathLength += std::sqrt(dx*dx + dy*dy);
            }
            std::cout << "Total path length: " << pathLength << std::endl;
        }
        {
            // Write covariance matrices along the path
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
        {
            // Write state times (timestamps) along the path
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
        {
            // Write the RRT tree edges to file
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
    }
    else if (choice == 2)
    {
        // ====== BENCHMARKING SECTION ======
        auto benchmark_ss = std::make_shared<oc::SimpleSetup>(si);
        benchmark_ss->setStartAndGoalStates(start, goal, 0.25);
        benchmarkCCRRT(benchmark_ss);
        return 0;
    }
    return 0;
}