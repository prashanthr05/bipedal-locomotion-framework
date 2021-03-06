/**
 * @file FloatingBaseSystemKinematics.h
 * @authors Giulio Romualdi
 * @copyright 2020 Istituto Italiano di Tecnologia (IIT). This software may be modified and
 * distributed under the terms of the GNU Lesser General Public License v2.1 or any later version.
 */

#ifndef BIPEDAL_LOCOMOTION_SYSTEM_FLOATING_BASE_SYSTEM_KINEMATICS_H
#define BIPEDAL_LOCOMOTION_SYSTEM_FLOATING_BASE_SYSTEM_KINEMATICS_H

#include <memory>
#include <vector>

#include <BipedalLocomotion/System/DynamicalSystem.h>
#include <BipedalLocomotion/System/ContactWrench.h>
#include <BipedalLocomotion/ParametersHandler/IParametersHandler.h>

#include <Eigen/Dense>

namespace BipedalLocomotion
{
namespace System
{

/**
 * FloatingBaseSystemKinematics describes a floating base system kinematics.
 * The FloatingBaseSystemKinematics inherits from a generic DynamicalSystem where:
 * - DynamicalSystem::StateType is described by an std::tuple containing:
 *   - Eigen::Vector6d: position of the base w.r.t. the inertial frame
 *   - Eigen::Matrix3d: rotation matrix \f${} ^ I R _ {b}\f$. Matrix that transform a vector
 * whose coordinates are expressed in the base frame in the inertial frame;
 *   - Eigen::VectorXd: the joint positions [in rad].
 * - DynamicalSystem::StateDerivativeType is described by an std::tuple containing:
 *   - Eigen::Vector6d: base velocity w.r.t. the inertial frame;
 *   - Eigen::Matrix3d: rate of change of the rotation matrix \f${} ^ I \dot{R} _ {b}\f$.
 * whose coordinates are expressed in the base frame in the inertial frame;
 *   - Eigen::VectorXd: the joint velocities [in rad/s].
 * - DynamicalSystem::InputType is described by an std::tuple containing:
 *   - Eigen::Vector6d: base twist w.r.t. the inertial frame;
 *   - Eigen::VectorXd: the joint velocities [in rad/s].
 */
class FloatingBaseSystemKinematics
    : public DynamicalSystem<std::tuple<Eigen::Vector3d, Eigen::Matrix3d, Eigen::VectorXd>,
                             std::tuple<Eigen::Vector3d, Eigen::Matrix3d, Eigen::VectorXd>,
                             std::tuple<Eigen::Matrix<double, 6, 1>, Eigen::VectorXd>>
{
    double m_rho{0.01}; /**< Regularization term used for the Baumgarte stabilization over the SO(3)
                           group */

public:
    /**
     * Initialize the Dynamical system.
     * @note Please call this function only if you want to set an arbitrary value for the parameter
     * used in the Baumgarte stabilization \f$\rho\f$ (The default value is 0.01 ). In this case the
     * handler should contain a key called rho.
     * @param handler pointer to the parameter handler.
     * @return true in case of success/false otherwise.
     */
    bool initalize(std::weak_ptr<ParametersHandler::IParametersHandler> handler) override;

    /**
     * Computes the floating based system dynamics. It return \f$f(x, u, t)\f$.
     * @note The control input and the state have to be set separately with the methods
     * setControlInput and setState.
     * @param time the time at witch the dynamics is computed.
     * @param stateDynamics tuple containing a reference to the element of the state derivative
     * @return true in case of success, false otherwise.
     */
    bool dynamics(const double& time, StateDerivativeType& stateDerivative) final;

    /**
     * Destructor.
     */
    ~FloatingBaseSystemKinematics() = default;
};

} // namespace System
} // namespace BipedalLocomotion

#endif // BIPEDAL_LOCOMOTION_SYSTEM_FLOATING_BASE_SYSTEM_KINEMATICS_H
