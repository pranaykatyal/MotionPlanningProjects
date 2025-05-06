// =======================================================================================
// MotionValidator.cpp
// Implements motion validation for Chance-Constrained RRT (CCRRT).
// This file provides:
//   - UncertaintyManager: Propagates and stores state uncertainty (mean/covariance/timestamp)
//   - CCRRTMotionValidator: Checks if a motion (edge) between two states is valid
//     considering static obstacles, dynamic obstacles, and chance constraints.
//
// Key Concepts:
// - Each state can have associated uncertainty (mean, covariance, timestamp).
// - Static obstacles: Fixed in space (collision checked at all times).
// - Dynamic obstacles: Move over time (collision checked at the correct time).
// - Chance constraints: Probabilistic collision avoidance using state uncertainty.
// =======================================================================================

#include "MotionValidator.h"
#include <ompl/control/SpaceInformation.h>
#include <cmath>

// Add these declarations
extern std::vector<Obstacle> staticObstacles;

void UncertaintyManager::storeUncertainty(const ob::State* state, 
    const Eigen::VectorXd& mean, const Eigen::MatrixXd& cov, double timestamp) 
{
    stateUncertainty_[state] = StateUncertainty{mean, cov, timestamp};
}

void UncertaintyManager::propagateUncertainty(const ob::State* from, const ob::State* to,
    const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
    const Eigen::VectorXd& control, const Eigen::MatrixXd& Pw,
    double deltaTime)
{
    auto it = stateUncertainty_.find(from);
    double currentTime = (it != stateUncertainty_.end()) ? 
        it->second.timestamp : 0.0;
    
    if (it == stateUncertainty_.end()) {
        int dim = A.rows();
        Eigen::VectorXd mean = Eigen::VectorXd::Zero(dim);
        const auto* rs = from->as<ob::RealVectorStateSpace::StateType>();
        for (int i = 0; i < dim; ++i)
            mean(i) = rs->values[i];
        stateUncertainty_[from] = StateUncertainty{mean, Pw, currentTime};
        it = stateUncertainty_.find(from);
    }

    // Account for relative motion in uncertainty propagation
    Eigen::VectorXd nextMean = A * it->second.mean + B * control;
    Eigen::MatrixXd nextCov = A * it->second.covariance * A.transpose() + Pw;
    
    // Add additional uncertainty due to dynamic obstacle motion
    // This increases uncertainty in the direction of obstacle motion
    Eigen::Matrix2d dynamicUncertainty = Eigen::Matrix2d::Zero();
    dynamicUncertainty(0,0) = 0.1; // Additional uncertainty in x direction due to moving obstacle
    dynamicUncertainty(1,1) = 0.1; // Add some in y as well if obstacle can move in y
    nextCov.block<2,2>(0,0) += dynamicUncertainty * deltaTime;
    stateUncertainty_[to] = StateUncertainty{nextMean, nextCov, currentTime + deltaTime};
}

bool UncertaintyManager::satisfiesChanceConstraints(const ob::State* state, 
    const std::vector<Obstacle>& obstacles) const
{
    auto it = stateUncertainty_.find(state);
    if (it == stateUncertainty_.end()) return true;
    const double currentTime = it->second.timestamp;
    Eigen::Vector2d mean = it->second.mean.head<2>();
    Eigen::Matrix2d cov = it->second.covariance.block<2,2>(0, 0);
    for (const auto& obs : obstacles) {
        if (obs.isCircular()) {
            if (!isCircleConstraintSatisfied(mean, cov, obs)) 
                return false;
        } else {
            // Rectangle: use circumscribed circle
            Eigen::Vector2d obsCenter = obs.getCenter();
            double obsRadius = std::hypot(obs.getWidth()/2, obs.getHeight()/2);
            Obstacle circObs(obsCenter[0], obsCenter[1], obsRadius);
            if (!isCircleConstraintSatisfied(mean, cov, circObs))
                return false;
        }
    }
    return true;
}

Eigen::Vector2d UncertaintyManager::getDynamicObstaclePosition(
    const Obstacle& obs, double time) const
{
    // For dynamic obstacles, compute position based on time
    // This should match the motion model in DynamicObstacle class
    if (!obs.isCircular()) {
        const double velocity = 1.0; // Matches velocity from main
        Eigen::Vector2d initialPos(obs.getX() + obs.getWidth()/2, obs.getY() + obs.getHeight()/2);
        Eigen::Vector2d displacement(velocity * time, 0.0);
        return initialPos + displacement;
    }
    return obs.getCenter();
}

bool UncertaintyManager::isCircleConstraintSatisfied(const Eigen::Vector2d& mean, 
    const Eigen::Matrix2d& cov, const Obstacle& obs) const
{
    Eigen::Vector2d diff = obs.getCenter() - mean;
    double dist = diff.norm();
    double directionalVar = (dist > 1e-6) ? 
        (diff.normalized().transpose() * cov * diff.normalized()) :
        cov.eigenvalues().real().maxCoeff();
    double sigma = std::sqrt(directionalVar);
    double beta = std::sqrt(2) * erfcinv(2 * psafe_ - 1);
    return dist > obs.getRadius() + beta * sigma;
}

bool UncertaintyManager::isRectConstraintSatisfied(const Eigen::Vector2d& mean, 
    const Eigen::Matrix2d& cov, const Obstacle& obs) const
{
    Eigen::Vector2d obsCenter(obs.getX() + obs.getWidth()/2, obs.getY() + obs.getHeight()/2);
    double obsRadius = std::hypot(obs.getWidth()/2, obs.getHeight()/2);
    return isCircleConstraintSatisfied(mean, cov, Obstacle(obsCenter[0], obsCenter[1], obsRadius));
}

// Implementation of CCRRTMotionValidator
CCRRTMotionValidator::CCRRTMotionValidator(const ob::SpaceInformationPtr& si, double psafe)
    : ob::MotionValidator(si), psafe_(psafe)
{
}

void CCRRTMotionValidator::setObstacles(const std::vector<Obstacle>& obstacles) {
    obstacles_ = obstacles;
}

void CCRRTMotionValidator::setStateUncertainty(std::map<StateKey, CCRRTDetail::StateWithCovariance>* stateUncertainty) {
    stateUncertainty_ = stateUncertainty;
}

bool CCRRTMotionValidator::checkMotion(const ob::State* s1, const ob::State* s2) const {
    double dist = si_->distance(s1, s2);
    unsigned int nd = std::max<unsigned int>(2, std::ceil(dist / si_->getStateValidityCheckingResolution()));
    ob::State *test = si_->allocState();

    for (unsigned int i = 1; i <= nd; ++i) {
        double ratio = (double)i / (double)nd;
        si_->getStateSpace()->interpolate(s1, s2, ratio, test);

        if (!si_->satisfiesBounds(test)) {
            si_->freeState(test);
            return false;
        }

        const auto *st = test->as<ob::RealVectorStateSpace::StateType>();
        Eigen::Vector2d stateVec(st->values[0], st->values[1]);

        double t = 0.0;
        if (stateUncertainty_) {
            StateKey key = StateKey::fromState(test);
            double minDist = std::numeric_limits<double>::max();
            for (const auto& pair : *stateUncertainty_) {
                double dx = key.x - pair.first.x;
                double dy = key.y - pair.first.y;
                double d = dx*dx + dy*dy;
                if (d < minDist) {
                    minDist = d;
                    t = pair.second.timestamp;
                }
            }
        }

        // Check static obstacles (rectangular collision)
        for (const auto& obs : obstacles_) {
            if (!obs.isCircular()) {
                double x = obs.getX(), y = obs.getY();
                double w = obs.getWidth(), h = obs.getHeight();
                if (stateVec[0] >= x && stateVec[0] <= x + w &&
                    stateVec[1] >= y && stateVec[1] <= y + h) {
                    si_->freeState(test);
                    return false;
                }
            } else {
                // Fallback for circles (should not occur)
                Eigen::Vector2d obsCenter = obs.getCenter();
                double obsRadius = obs.getRadius();
                double d = (stateVec - obsCenter).norm();
                if (d <= obsRadius) {
                    si_->freeState(test);
                    return false;
                }
            }
        }

        // Chance constraint check for all obstacles (static only)
        if (stateUncertainty_) {
            StateKey key = StateKey::fromState(test);
            const CCRRTDetail::StateWithCovariance* unc = nullptr;
            double minDist = std::numeric_limits<double>::max();
            for (const auto& pair : *stateUncertainty_) {
                double dx = key.x - pair.first.x;
                double dy = key.y - pair.first.y;
                double d = dx*dx + dy*dy;
                if (d < minDist) {
                    minDist = d;
                    unc = &pair.second;
                }
            }
            if (unc) {
                // Static obstacles
                for (const auto& obs : obstacles_) {
                    if (!obs.isCircular()) {
                        if (!isChanceConstraintSatisfied(*unc, obs)) {
                            si_->freeState(test);
                            return false;
                        }
                    } else {
                        // Fallback for circles (should not occur)
                        if (!isChanceConstraintSatisfied(*unc, obs)) {
                            si_->freeState(test);
                            return false;
                        }
                    }
                }
            }
        }
    }
    si_->freeState(test);
    return true;
}

bool CCRRTMotionValidator::checkMotion(const ob::State* s1, const ob::State* s2,
    std::pair<ob::State*, double>& /*lastValid*/) const
{
    return checkMotion(s1, s2);
}