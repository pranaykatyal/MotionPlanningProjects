// =======================================================================================
// MotionValidator.h
// Header for motion validation in Chance-Constrained RRT (CCRRT).
// Defines:
//   - StateKey: Helper for identifying states by position.
//   - CCRRTDetail::StateWithCovariance: Stores mean, covariance, and timestamp.
//   - UncertaintyManager: Propagates and stores uncertainty for each state.
//   - CCRRTMotionValidator: Checks if a motion is valid, considering:
//         1. Static obstacles (fixed in space)
//         2. Dynamic obstacles (move over time)
//         3. Chance constraints (probabilistic collision using uncertainty)
//
// State: 2D position (x, y)
// Control: 2D velocity (vx, vy)
// =======================================================================================

#ifndef MOTION_VALIDATOR_H
#define MOTION_VALIDATOR_H

#pragma once
#include <ompl/control/SpaceInformation.h>
#include <ompl/base/MotionValidator.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <Eigen/Dense>
#include <vector>
#include "Obstacles.h"

inline double erfcinv(double x) {
    if (x >= 2.0) return -std::numeric_limits<double>::infinity();
    if (x <= 0.0) return std::numeric_limits<double>::infinity();

    const double pp = (x < 1.0) ? x : 2.0 - x;
    const double t = std::sqrt(-2.0 * std::log(pp / 2.0));
    double p = -0.70711 * ((2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t);

    for (int i = 0; i < 2; i++) {
        double err = std::erfc(p) - pp;
        p += err / (1.12837916709551257 * std::exp(-p * p) - p * err);
    }
    return (x < 1.0) ? p : -p;
}

namespace ob = ompl::base;
namespace oc = ompl::control;

struct StateKey {
    double x, y;
    bool operator<(const StateKey& other) const {
        if (x != other.x) return x < other.x;
        return y < other.y;
    }
    static StateKey fromState(const ob::State* s) {
        auto* st = s->as<ob::RealVectorStateSpace::StateType>();
        return {st->values[0], st->values[1]};
    }
};

namespace CCRRTDetail {
    struct StateWithCovariance {
        Eigen::VectorXd mean;
        Eigen::MatrixXd covariance;
        double timestamp;
    };
}

class UncertaintyManager {
public:
    UncertaintyManager(double psafe);
    void storeUncertainty(const ob::State* state,
                          const Eigen::VectorXd& mean, const Eigen::MatrixXd& cov, double timestamp);
    void propagateUncertainty(const ob::State* from, const ob::State* to,
                              const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                              const Eigen::VectorXd& control, const Eigen::MatrixXd& Pw,
                              double deltaTime);
    bool satisfiesChanceConstraints(const ob::State* state,
                                   const std::vector<Obstacle>& obstacles) const;
    struct StateUncertainty {
        Eigen::VectorXd mean;
        Eigen::MatrixXd covariance;
        double timestamp;
    };
    const std::map<const ob::State*, StateUncertainty>& getStateUncertainty() const;
private:
    bool isCircleConstraintSatisfied(const Eigen::Vector2d& mean,
                                    const Eigen::Matrix2d& cov, const Obstacle& obs) const;
    bool isRectConstraintSatisfied(const Eigen::Vector2d& mean,
                                   const Eigen::Matrix2d& cov, const Obstacle& obs) const;
    Eigen::Vector2d getDynamicObstaclePosition(const Obstacle& obs, double time) const;
    double psafe_;
    std::map<const ob::State*, StateUncertainty> stateUncertainty_;
};

class CCRRTMotionValidator : public ob::MotionValidator {
public:
    CCRRTMotionValidator(const ob::SpaceInformationPtr& si, double psafe);
    void setObstacles(const std::vector<Obstacle>& obstacles);
    void setStateUncertainty(std::map<StateKey, CCRRTDetail::StateWithCovariance>* stateUncertainty);
    bool checkMotion(const ob::State* s1, const ob::State* s2) const override;
    bool checkMotion(const ob::State* s1, const ob::State* s2,
        std::pair<ob::State*, double>& lastValid) const override;
private:
    double psafe_;
    std::vector<Obstacle> obstacles_;
    std::map<StateKey, CCRRTDetail::StateWithCovariance>* stateUncertainty_ = nullptr;
    bool isChanceConstraintSatisfied(const CCRRTDetail::StateWithCovariance& stateUnc,
                                     const Obstacle& obs) const {
        Eigen::Vector2d mean = stateUnc.mean.head<2>();
        Eigen::Matrix2d cov = stateUnc.covariance.block<2,2>(0, 0);
        if (!obs.isCircular()) {
            Eigen::Vector2d obsCenter = obs.getCenter();
            double obsRadius = std::hypot(obs.getWidth()/2, obs.getHeight()/2);
            Eigen::Vector2d diff = obsCenter - mean;
            double dist = diff.norm();
            double directionalVar = (dist > 1e-6) ?
                (diff.normalized().transpose() * cov * diff.normalized()) :
                cov.eigenvalues().real().maxCoeff();
            double sigma = std::sqrt(directionalVar);
            double beta = std::sqrt(2) * erfcinv(2 * (1 - psafe_));
            return dist > obsRadius + beta * sigma;
        } else {
            Eigen::Vector2d diff = obs.getCenter() - mean;
            double dist = diff.norm();
            double directionalVar = (dist > 1e-6) ?
                (diff.normalized().transpose() * cov * diff.normalized()) :
                cov.eigenvalues().real().maxCoeff();
            double sigma = std::sqrt(directionalVar);
            double beta = std::sqrt(2) * erfcinv(2 * (1 - psafe_));
            return dist > obs.getRadius() + beta * sigma;
        }
    }
};

#endif // MOTION_VALIDATOR_H