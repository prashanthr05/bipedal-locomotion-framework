/**
 * @file DILIGENT.cpp
 * @authors Prashanth Ramadoss
 * @copyright 2020 Istituto Italiano di Tecnologia (IIT). This software may be modified and
 * distributed under the terms of the GNU Lesser General Public License v2.1 or any later version.
 */

#include <BipedalLocomotion/FloatingBaseEstimators/DILIGENT.h>
#include <BipedalLocomotion/FloatingBaseEstimators/FloatingBaseExtendedKinematicsLieGroup.h>
#include <iDynTree/Core/EigenHelpers.h>

#include <manif/manif.h>

using namespace BipedalLocomotion::Estimators;

class DILIGENT::Impl
{
public:
    bool extractStateFromLieGroup(const FloatingBaseExtendedKinematicsLieGroup& G,
                                  const bool& estimateBias,
                                  FloatingBaseEstimators::InternalState& state);

    void constructStateLieGroup(const FloatingBaseEstimators::InternalState& state,
                                const bool& estimateBias,
                                FloatingBaseExtendedKinematicsLieGroup& G);

    /**
     * Construct  the state covariance matrix from the
     * internal state standard deviation object,
     */
    void constuctStateVar(const FloatingBaseEstimators::StateStdDev& stateStdDev,
                          const bool& estimateBias,
                          Eigen::MatrixXd& P);

    /**
     * Extract internal state standard deviation object,
     * from the diagonal elements of the state covariance matrix
     */
    void extractStateVar(const Eigen::MatrixXd& P,
                         const FloatingBaseExtendedKinematicsLieGroupTangent& Omegak,
                         const bool& estimateBias,
                         FloatingBaseEstimators::StateStdDev& stateStdDev);

    /**
     * Compute left parametrized velocity
     */
    void calcOmegak(const FloatingBaseEstimators::InternalState& X,
                    const FloatingBaseEstimators::Measurements& meas,
                    const double& dt,
                    const Eigen::Vector3d& g,
                    const bool& estimateBias,
                    FloatingBaseExtendedKinematicsLieGroupTangent& Omegak);

    /**
     * Propagate internal state (mean) of the estimator
     * using Lie group motion integration
     */
    bool propagateStates(FloatingBaseExtendedKinematicsLieGroup& Xk,
                         const FloatingBaseExtendedKinematicsLieGroupTangent& Omegak,
                         const bool& estimateBias,
                         FloatingBaseEstimators::InternalState& state);

    /**
     * Perform the Kalman filter update step given measurements and Jacobians
     */
    bool updateStates(const Eigen::VectorXd& deltaY,
                      const Eigen::MatrixXd measModelJacobian,
                      const Eigen::MatrixXd& measNoiseVar,
                      FloatingBaseEstimators::InternalState& state,
                      Eigen::MatrixXd& P);


    /**
     * Compute continuous time system noise covariance matrix
     * using the predicted internal state estimates
     */
    void calcQk(const FloatingBaseEstimators::SensorsStdDev& sensStdDev,
                const FloatingBaseEstimators::InternalState& state,
                const FloatingBaseExtendedKinematicsLieGroupTangent& Omegak,
                const size_t dim,
                const double& dt,
                const bool& estimateBias,
                Eigen::MatrixXd& Qc);

    /**
     * Compute left parametrized velocity Jacobian
     */
    void calcSlantFk(const FloatingBaseEstimators::InternalState& X,
                     const size_t dim,
                     const double& dt,
                     const Eigen::Vector3d& g,
                     const bool& estimateBias,
                     Eigen::MatrixXd& slantFk);
    
    std::size_t dimensions(const FloatingBaseEstimators::InternalState& state, 
                           const bool& estimateBias)
    {
        std::size_t dim{9};
        dim += motionDim*state.supportFrameData.size();
        dim += motionDim*state.landmarkData.size();
        
        if (estimateBias)
        {
            dim += 6;
        }
        return dim;        
    }
    
    const std::size_t imuPositionOffset{0};
    const std::size_t imuOrientationOffset{3};
    const std::size_t imuLinearVelOffset{6};
    const std::size_t extMotionDim{9};
    const std::size_t motionDim{6};
    const std::size_t accBiasOffsetFromTail{6};
    const std::size_t gyroBiasOffsetFromTail{3};
    Eigen::Matrix3d I3;
    manif::SE3Tangentd zeroTwist;
    
    Eigen::MatrixXd m_P;      /**< state covariance matrix */
    FloatingBaseExtendedKinematicsLieGroup Xk;
    FloatingBaseExtendedKinematicsLieGroupTangent Omegak;
    FloatingBaseExtendedKinematicsLieGroupTangent negOmegak;
    Eigen::MatrixXd slantFk, Qk;
    Eigen::MatrixXd AdjNegativeOmegaLifted;
    Eigen::MatrixXd Jr; /**<right Jacobian of Lie group at Omega as tangent */
    Eigen::MatrixXd Fk; /**< system dynamics Jacobian*/
    
    /**< for proper handling of contact frame index and 
     * landmark index within the same map, 
     * if landmark id = 1 and landmarkIDsOffset = 1000, 
     * in the LieGroup object support frame map,
     * it will be internally stored as 1001
     */
    int landmarkIDsOffset{1000}; 
};

DILIGENT::DILIGENT() : m_pimpl(std::make_unique<Impl>())
{
    m_state.imuOrientation.setIdentity();
    m_state.imuPosition.setZero();
    m_state.imuLinearVelocity.setZero();

    m_state.accelerometerBias.setZero();
    m_state.gyroscopeBias.setZero();

    m_statePrev = m_state;
    m_estimatorOut.state = m_state;

    m_meas.acc.setZero();
    m_meas.gyro.setZero();

    m_measPrev = m_meas;

    m_stateStdDev.imuOrientation.setZero();
    m_stateStdDev.imuPosition.setZero();
    m_stateStdDev.imuLinearVelocity.setZero();

    m_stateStdDev.accelerometerBias.setZero();
    m_stateStdDev.gyroscopeBias.setZero();

    m_priors = m_stateStdDev;
    m_estimatorOut.stateStdDev = m_stateStdDev;

    m_sensorsDev.gyroscopeNoise.setZero();
    m_sensorsDev.accelerometerNoise.setZero();
    m_sensorsDev.accelerometerBiasNoise.setZero();
    m_sensorsDev.gyroscopeBiasNoise.setZero();
    m_sensorsDev.contactFootLinvelNoise.setZero();
    m_sensorsDev.contactFootAngvelNoise.setZero();
    m_sensorsDev.swingFootLinvelNoise.setZero();
    m_sensorsDev.swingFootAngvelNoise.setZero();
    m_sensorsDev.landmarkMeasurementNoise.setZero();
    m_sensorsDev.landmarkPredictionNoise.setZero();
    
    m_pimpl->I3 = Eigen::Matrix3d::Identity();
    m_pimpl->zeroTwist.setZero();
}

DILIGENT::~DILIGENT() = default;

bool DILIGENT::customInitialization(std::weak_ptr<BipedalLocomotion::ParametersHandler::IParametersHandler> handler)
{
    auto handle = handler.lock();
    if (handle == nullptr)
    {
        std::cerr << "[DILIGENT::customInitialization] The parameter handler has expired. Please check its scope."
        << std::endl;
        return false;
    }

    // setup options related entities
    auto optionsHandle = handle->getGroup("Options");
    if (!setupOptions(optionsHandle))
    {
        std::cerr << "[DILIGENT::customInitialization] Could not load options related parameters."
        << std::endl;
        return false;
    }

    // setup sensor standard deviations
    auto sensorDevHandle = handle->getGroup("SensorsStdDev");
    if (!setupSensorDevs(sensorDevHandle))
    {
        std::cerr << "[DILIGENT::customInitialization] Could not load sensor stddev related parameters."
        << std::endl;
        return false;
    }

    // setup initial states
    auto initStateHandle = handle->getGroup("InitialStates");
    if (!setupInitialStates(initStateHandle))
    {
        std::cerr << "[DILIGENT::customInitialization] Could not load initial states related parameters."
        << std::endl;
        return false;
    }

    // setup initial state standard deviations
    auto priorDevHandle = handle->getGroup("PriorsStdDev");
    if (!setupPriorDevs(priorDevHandle))
    {
        std::cerr << "[DILIGENT::customInitialization] Could not load prior stddev related parameters."
        << std::endl;
        return false;
    }


    m_pimpl->constuctStateVar(m_priors, m_options.imuBiasEstimationEnabled, m_pimpl->m_P); // construct priors
    return true;
}

bool DILIGENT::resetEstimator(const FloatingBaseEstimators::InternalState& newState,
                                               const FloatingBaseEstimators::StateStdDev& newPriorDev)
{
    m_state = newState;
    m_stateStdDev = newPriorDev;
    m_priors = newPriorDev;

    m_pimpl->constuctStateVar(m_priors, m_options.imuBiasEstimationEnabled, m_pimpl->m_P); // construct priors
    return true;
}

bool DILIGENT::resetEstimator(const FloatingBaseEstimators::InternalState& newState,
                              const FloatingBaseEstimators::StateStdDev& newPriorDev,
                              const FloatingBaseEstimators::SensorsStdDev& newSensorsDev)
{
    m_sensorsDev = newSensorsDev;
    resetEstimator(newState, newPriorDev);
    return true;
}

bool DILIGENT::predictState(const FloatingBaseEstimators::Measurements& meas,
                            const double& dt)
{
    auto dim = m_pimpl->dimensions(m_state, m_options.imuBiasEstimationEnabled);    
    m_pimpl->calcOmegak(m_state, meas, dt,
                        m_options.accelerationDueToGravity,
                        m_options.imuBiasEstimationEnabled,
                        m_pimpl->Omegak);
    
    m_pimpl->negOmegak = -m_pimpl->Omegak;
    
    // left parametrized velocity Jacobian
    m_pimpl->calcSlantFk(m_state, dim, dt,
                         m_options.accelerationDueToGravity,
                         m_options.imuBiasEstimationEnabled, m_pimpl->slantFk); // compute slantFk at priori state 
    
    // prediction model covariance matrix
    m_pimpl->calcQk(m_sensorsDev, m_state, 
                    m_pimpl->Omegak, dim, dt, 
                    m_options.imuBiasEstimationEnabled, m_pimpl->Qk); // compute  Qk at priori state and previous measure

    // Map the changes from tangent space at X to tangent space at Identity
    m_pimpl->AdjNegativeOmegaLifted = m_pimpl->negOmegak.exp().adj();
    m_pimpl->Jr = m_pimpl->Omegak.rjac(); // right Jacobian of Lie group at Omega as tangent
    
    if (m_pimpl->AdjNegativeOmegaLifted.cols() != m_pimpl->Jr.cols() &&
        m_pimpl->Jr.cols() != m_pimpl->slantFk.rows())
    {
        std::cerr << "[DILIGENT::predictState] Jacobian matrices size mismatch" << std::endl;
        return false;        
    }
    
    m_pimpl->Fk = m_pimpl->AdjNegativeOmegaLifted + (m_pimpl->Jr*m_pimpl->slantFk);
        
    // m_state is now predicted state after this function call
    if (!m_pimpl->propagateStates(m_pimpl->Xk, 
                                  m_pimpl->Omegak, 
                                  m_options.imuBiasEstimationEnabled, 
                                  m_state))
    {
        return false;
    }
    
    if (m_pimpl->Fk.cols() != m_pimpl->m_P.rows())
    {
        std::cerr << "[DILIGENT::predictState] covariance matrices size mismatch" << std::endl;
        return false;        
    }
    
    m_pimpl->m_P = m_pimpl->Fk*m_pimpl->m_P*(m_pimpl->Fk.transpose()) + 
                   m_pimpl->Jr*(m_pimpl->Qk*dt)*(m_pimpl->Jr.transpose());
//     m_pimpl->extractStateVar(m_pimpl->m_P,m_options.imuBiasEstimationEnabled, m_stateStdDev); // unwrap state covariance matrix diagonal*/

    return true;
}

bool DILIGENT::updateKinematics(const FloatingBaseEstimators::Measurements& meas,
                                const double& dt)
{
//     Eigen::VectorXd deltaY;
//     Eigen::MatrixXd measModelJacobian, measNoiseVar;
// 
//     // extract current predicted state
//     Eigen::Matrix3d A_R_IMU = m_state.imuOrientation.toRotationMatrix();
//     Eigen::Matrix3d A_R_LF = m_state.lContactFrameOrientation.toRotationMatrix();
//     Eigen::Matrix3d A_R_RF = m_state.rContactFrameOrientation.toRotationMatrix();
// 
//     const Eigen::Vector3d& p = m_state.imuPosition;
//     const Eigen::Vector3d& plf = m_state.lContactFramePosition;
//     const Eigen::Vector3d& prf = m_state.rContactFramePosition;
// 
//     iDynTree::JointPosDoubleArray jointPos(meas.encoders.size());
//     iDynTree::toEigen(jointPos) = meas.encoders;
//     iDynTree::Transform yIMU_H_LF, yIMU_H_RF;
//     iDynTree::MatrixDynSize LF_J_IMULF, RF_J_IMURF;
//     m_modelComp.getIMU_H_feet(jointPos, yIMU_H_LF, yIMU_H_RF);
//     m_modelComp.getLeftTrivializedJacobianFeetWRTIMU(jointPos, LF_J_IMULF, RF_J_IMURF);
// 
//     auto Jl{iDynTree::toEigen(LF_J_IMULF)};
//     auto Jr{iDynTree::toEigen(RF_J_IMURF)};
// 
//     Eigen::VectorXd encodersVar = m_sensorsDev.encodersNoise.array().square();
//     Eigen::MatrixXd Renc = static_cast<Eigen::MatrixXd>(encodersVar.asDiagonal());
// 
//     // For double support, the measurement model is as follows
//     //
//     // h(X) = [A_R_B.T A_R_LF  A_R_B.T (plf - p)                                    ]
//     //        [                                1                                    ]
//     //        [                                   A_R_B.T A_R_RF   A_R_B.T (prf - p)]
//     //        [                                                                    1]
//     //
//     // h(X) \in SE(3) \times SE(3)
//     // hinv = h^{-1}(x) can be easily computed similar to inverse of a spatial transform
//     // by exploiting the concept of direct product induced by composite manifolds.
//     // Here we have a direct product of rigid body transforms SE(3) forming the composite manifold
//     // SE(3)xSE(3)
//     //
//     // The innovation term in the Lie algebra of the SE(3) \times SE(3) Liegroup space becomes,
//     // deltaY = logvee_SE3xSE3(hinv, y)
//     //
//     // The measurement model Jacobian for the double support, if biases are included,
//     // H = [-A_R_LF.T A_R_IMU | -A_R_LF.T [p - plf]x A_R_IMU |  0_3 |   I | 0_3 | 0_3 | 0_3 | 0_3 | 0_3 ]
//     //     [              0_3 |            -A_R_LF.T A_R_IMU |  0_3 | 0_3 |   I | 0_3 | 0_3 | 0_3 | 0_3 ]
//     //     [-A_R_RF.T A_R_IMU | -A_R_RF.T [p - plf]x A_R_IMU |  0_3 | 0_3 | 0_3 |   I | 0_3 | 0_3 | 0_3 ]
//     //     [              0_3 |            -A_R_RF.T A_R_IMU |  0_3 | 0_3 | 0_3 | 0_3 |   I | 0_3 | 0_3 ]
//     // I is the 3d identity matrix, []x is the 3d skew symmetric matrix
//     // If biases are enabled, the last 6 columns are not considered
//     //
//     // The measurement noise covariance,
//     // Rc = blkdiag(LF_J_{IMU,LF} Renc LF_J_{IMU,LF}.T,  RF_J_{IMU,RF} Renc RF_J_{IMU,RF}.T)
//     // Rk = Rc/dt is the discretized measurement noise covariance matrix
//     // The measurement noise is left-trivialized forward kinematic velocity noise which is computed
//     // using the manipulator Jacobian of the foot with respect to the IMU
//     //
//     // These matrices can easily be reduced for single support cases where observation space is defined by SE(3) Lie group
// 
//     if (meas.lfInContact && meas.rfInContact)
//     {
//         const int measurementSpaceDims{12};
// 
//         // prepare measurement model Jacobian H
//         if (m_options.imuBiasEstimationEnabled)
//         {
//             measModelJacobian.resize(measurementSpaceDims, m_pimpl->m_vecSizeWBias);
//         }
//         else
//         {
//             measModelJacobian.resize(measurementSpaceDims, m_pimpl->m_vecSizeWOBias);
//         }
// 
//         // prepare measurement error in tangent space
//         deltaY.resize(measurementSpaceDims);
//         Pose yLF = m_pimpl->iDynPose2manifPose(yIMU_H_LF);
//         Pose hLF = m_pimpl->eigenPose2manifPose(A_R_IMU.transpose()*A_R_LF, A_R_IMU.transpose()*(plf - p));
//         auto lfPoseError = yLF - hLF; // this performs logvee_SE3(inv(hLF), yLF)
// 
//         Pose yRF = m_pimpl->iDynPose2manifPose(yIMU_H_RF);
//         Pose hRF = m_pimpl->eigenPose2manifPose(A_R_IMU.transpose()*A_R_RF, A_R_IMU.transpose()*(prf - p));
//         auto rfPoseError = yRF - hRF; // this performs logvee_SE3(inv(hLF), yLF)
//         deltaY << lfPoseError.coeffs(), rfPoseError.coeffs();
// 
//         // just aliasing using a reference
//         Eigen::MatrixXd& H = measModelJacobian;
// 
//         H.block<3, 3>(0, m_pimpl->m_vecOffsets.imuPosition) = -A_R_LF.transpose()*A_R_IMU;
//         H.block<3, 3>(0, m_pimpl->m_vecOffsets.imuOrientation) = -A_R_LF.transpose()*iDynTree::skew(p - plf)*A_R_IMU;
//         H.block<3, 3>(0, m_pimpl->m_vecOffsets.imuLinearVel).setZero();
//         H.block<3, 3>(0, m_pimpl->m_vecOffsets.lContactFramePosition).setIdentity();
//         H.block<3, 9>(0, m_pimpl->m_vecOffsets.lContactFrameOrientation).setZero();
// 
//         H.block<3, 3>(3, m_pimpl->m_vecOffsets.imuPosition).setZero();
//         H.block<3, 3>(3, m_pimpl->m_vecOffsets.imuOrientation) = -A_R_LF.transpose()*A_R_IMU;
//         H.block<3, 6>(3, m_pimpl->m_vecOffsets.imuLinearVel).setZero();
//         H.block<3, 3>(3, m_pimpl->m_vecOffsets.lContactFrameOrientation).setIdentity();
//         H.block<3, 6>(3, m_pimpl->m_vecOffsets.rContactFramePosition).setZero();
// 
//         H.block<3, 3>(6, m_pimpl->m_vecOffsets.imuPosition) = -A_R_RF.transpose()*A_R_IMU;
//         H.block<3, 3>(6, m_pimpl->m_vecOffsets.imuOrientation) = -A_R_RF.transpose()*iDynTree::skew(p - prf)*A_R_IMU;
//         H.block<3, 9>(6, m_pimpl->m_vecOffsets.imuLinearVel).setZero();
//         H.block<3, 3>(6, m_pimpl->m_vecOffsets.rContactFramePosition).setIdentity();
//         H.block<3, 3>(6, m_pimpl->m_vecOffsets.rContactFrameOrientation).setZero();
// 
//         H.block<3, 3>(9, m_pimpl->m_vecOffsets.imuPosition).setZero();
//         H.block<3, 3>(9, m_pimpl->m_vecOffsets.imuOrientation) = -A_R_RF.transpose()*A_R_IMU;
//         H.block<3, 12>(9, m_pimpl->m_vecOffsets.imuLinearVel).setZero();
//         H.block<3, 3>(9, m_pimpl->m_vecOffsets.rContactFrameOrientation).setIdentity();
// 
//         if (m_options.imuBiasEstimationEnabled)
//         {
//             H.block<12, 6>(0, m_pimpl->m_vecOffsets.accBias).setZero();
//         }
// 
// 
//         // prepare measurement noise covariance R
//         measNoiseVar.resize(measurementSpaceDims, measurementSpaceDims);
//         measNoiseVar.topLeftCorner<6, 6>() = Jl*Renc*(Jl.transpose());
//         measNoiseVar.topRightCorner<6, 6>().setZero();
//         measNoiseVar.bottomRightCorner<6, 6>() = Jr*Renc*(Jr.transpose());
//         measNoiseVar.bottomLeftCorner<6, 6>().setZero();
//         measNoiseVar /= dt;
//     }
//     else if (meas.lfInContact)
//     {
//         const int measurementSpaceDims{6};
// 
//         // prepare measurement model Jacobian H
//         if (m_options.imuBiasEstimationEnabled)
//         {
//             measModelJacobian.resize(measurementSpaceDims, m_pimpl->m_vecSizeWBias);
//         }
//         else
//         {
//             measModelJacobian.resize(measurementSpaceDims, m_pimpl->m_vecSizeWOBias);
//         }
// 
//         // prepare measurement error in tangent space
//         deltaY.resize(measurementSpaceDims);
//         Pose yLF = m_pimpl->iDynPose2manifPose(yIMU_H_LF);
//         Pose hLF = m_pimpl->eigenPose2manifPose(A_R_IMU.transpose()*A_R_LF, A_R_IMU.transpose()*(plf - p));
//         auto lfPoseError = yLF - hLF; // this performs logvee_SE3(inv(hLF), yLF)
//         deltaY << lfPoseError.coeffs();
// 
//         // just aliasing using a reference
//         Eigen::MatrixXd& H = measModelJacobian;
// 
//         H.block<3, 3>(0, m_pimpl->m_vecOffsets.imuPosition) = -A_R_LF.transpose()*A_R_IMU;
//         H.block<3, 3>(0, m_pimpl->m_vecOffsets.imuOrientation) = -A_R_LF.transpose()*iDynTree::skew(p - plf)*A_R_IMU;
//         H.block<3, 3>(0, m_pimpl->m_vecOffsets.imuLinearVel).setZero();
//         H.block<3, 3>(0, m_pimpl->m_vecOffsets.lContactFramePosition).setIdentity();
//         H.block<3, 9>(0, m_pimpl->m_vecOffsets.lContactFrameOrientation).setZero();
// 
//         H.block<3, 3>(3, m_pimpl->m_vecOffsets.imuPosition).setZero();
//         H.block<3, 3>(3, m_pimpl->m_vecOffsets.imuOrientation) = -A_R_LF.transpose()*A_R_IMU;
//         H.block<3, 6>(3, m_pimpl->m_vecOffsets.imuLinearVel).setZero();
//         H.block<3, 3>(3, m_pimpl->m_vecOffsets.lContactFrameOrientation).setIdentity();
//         H.block<3, 6>(3, m_pimpl->m_vecOffsets.rContactFramePosition).setZero();
// 
//         if (m_options.imuBiasEstimationEnabled)
//         {
//             H.block<6, 6>(0, m_pimpl->m_vecOffsets.accBias).setZero();
//         }
// 
// 
//         // prepare measurement noise covariance R
//         measNoiseVar.resize(measurementSpaceDims, measurementSpaceDims);
//         measNoiseVar = Jl*Renc*(Jl.transpose());
// 
//         measNoiseVar /= dt;
//     }
//     else if (meas.rfInContact)
//     {
//         const int measurementSpaceDims{6};
// 
//         // prepare measurement model Jacobian H
//         if (m_options.imuBiasEstimationEnabled)
//         {
//             measModelJacobian.resize(measurementSpaceDims, m_pimpl->m_vecSizeWBias);
//         }
//         else
//         {
//             measModelJacobian.resize(measurementSpaceDims, m_pimpl->m_vecSizeWOBias);
//         }
// 
//         // prepare measurement error in tangent space
//         deltaY.resize(measurementSpaceDims);
//         Pose yRF = m_pimpl->iDynPose2manifPose(yIMU_H_RF);
//         Pose hRF = m_pimpl->eigenPose2manifPose(A_R_IMU.transpose()*A_R_RF, A_R_IMU.transpose()*(prf - p));
//         Twist rfPoseError = yRF - hRF; // this performs logvee_SE3(inv(hLF), yLF)
//         deltaY << rfPoseError.coeffs();
// 
//         // just aliasing using a reference
//         Eigen::MatrixXd& H = measModelJacobian;
//         H.block<3, 3>(0, m_pimpl->m_vecOffsets.imuPosition) = -A_R_RF.transpose()*A_R_IMU;
//         H.block<3, 3>(0, m_pimpl->m_vecOffsets.imuOrientation) = -A_R_RF.transpose()*iDynTree::skew(p - prf)*A_R_IMU;
//         H.block<3, 9>(0, m_pimpl->m_vecOffsets.imuLinearVel).setZero();
//         H.block<3, 3>(0, m_pimpl->m_vecOffsets.rContactFramePosition).setIdentity();
//         H.block<3, 3>(0, m_pimpl->m_vecOffsets.rContactFrameOrientation).setZero();
// 
//         H.block<3, 3>(3, m_pimpl->m_vecOffsets.imuPosition).setZero();
//         H.block<3, 3>(3, m_pimpl->m_vecOffsets.imuOrientation) = -A_R_RF.transpose()*A_R_IMU;
//         H.block<3, 12>(3, m_pimpl->m_vecOffsets.imuLinearVel).setZero();
//         H.block<3, 3>(3, m_pimpl->m_vecOffsets.rContactFrameOrientation).setIdentity();
// 
//         if (m_options.imuBiasEstimationEnabled)
//         {
//             H.block<6, 6>(0, m_pimpl->m_vecOffsets.accBias).setZero();
//         }
// 
//         // prepare measurement noise covariance R
//         measNoiseVar.resize(measurementSpaceDims, measurementSpaceDims);
//         measNoiseVar = Jr*Renc*(Jr.transpose());
//         measNoiseVar /= dt;
//     }
// 
//     if (meas.lfInContact || meas.rfInContact)
//     {
//         if (!m_pimpl->updateStates(deltaY, measModelJacobian, measNoiseVar, m_state, m_pimpl->m_P))
//         {
//             return false;
//         }
//         m_pimpl->extractStateVar(m_pimpl->m_P,m_options.imuBiasEstimationEnabled, m_stateStdDev); // unwrap state covariance matrix diagonal
//     }

    return true;
}

bool DILIGENT::updateLandmarkRelativePoses(const FloatingBaseEstimators::Measurements& meas,
                                           const double& dt)
{
    return true;
}

bool DILIGENT::Impl::propagateStates(FloatingBaseExtendedKinematicsLieGroup& Xk,
                                     const FloatingBaseExtendedKinematicsLieGroupTangent& Omegak,
                                     const bool& estimateBias,
                                     FloatingBaseEstimators::InternalState& state)
{
    auto exphatOmega = Omegak.exp();
    constructStateLieGroup(state, estimateBias, Xk);
    
    if (Xk.nrOfSupportFrames() != exphatOmega.nrOfSupportFrames())
    {
        std::cerr << "[DILIGENT::propagateStates] support frames size mismatch" << std::endl;
        return false;        
    }
    
    if (Xk.isAugmentedVectorUsed() != exphatOmega.isAugmentedVectorUsed())
    {
        std::cerr << "[DILIGENT::propagateStates] bias handling mismatch" << std::endl;
        return false;
    }
    
    auto M = Xk*exphatOmega;

    if (!extractStateFromLieGroup(M, estimateBias, state))
    {
        std::cerr << "[DILIGENT::propagateStates] unable to extract state from Lie group" << std::endl;
        return false;
    }

    return true;
}

bool DILIGENT::Impl::updateStates(const Eigen::VectorXd& deltaY,
                                  const Eigen::MatrixXd measModelJacobian,
                                  const Eigen::MatrixXd& measNoiseVar,
                                  FloatingBaseEstimators::InternalState& state,
                                  Eigen::MatrixXd& P)
{
//     if (measModelJacobian.cols() != P.rows())
//     {
//         std::cerr << "[DILIGENT::updateStates] Measurement model Jacobian size mismatch" << std::endl;
//         return false;
//     }
// 
//     if (measModelJacobian.rows() != measNoiseVar.rows())
//     {
//         std::cerr << "[DILIGENT::updateStates] Measurement noise covariance matrix size mismatch" << std::endl;
//         return false;
//     }
// 
//     bool estimateBias;
//     if (P.rows() == m_vecSizeWBias)
//     {
//         estimateBias = true;
//     }
//     else
//     {
//         estimateBias = false;
//     }
// 
//     Eigen::MatrixXd PHT = P*measModelJacobian.transpose();
//     Eigen::MatrixXd S = measModelJacobian*PHT + measNoiseVar;
//     Eigen::MatrixXd K = PHT*(S.inverse());
// 
//     Eigen::VectorXd deltaX = K*deltaY;
// 
//     // update estimate
//     if (!propagateStates(deltaX, estimateBias, state))
//     {
//         return false;
//     }
// 
//     // update covariance
//     IMUBipedMatrixLieGroupTangent v(deltaX);
//     Eigen::MatrixXd Jr = v.rjac();
//     auto IminusKH = Eigen::MatrixXd::Identity(P.rows(), P.cols()) - K*measModelJacobian;
//     P = Jr*IminusKH*P*(Jr.transpose());

    return true;
}

void DILIGENT::Impl::extractStateVar(const Eigen::MatrixXd& P,
                                     const FloatingBaseExtendedKinematicsLieGroupTangent& Omegak,
                                     const bool& estimateBias,
                                     FloatingBaseEstimators::StateStdDev& stateStdDev)
{
    stateStdDev.imuPosition =  P.block<3, 3>(imuPositionOffset, imuPositionOffset).diagonal().array().sqrt();
    stateStdDev.imuOrientation =  P.block<3, 3>(imuOrientationOffset, imuOrientationOffset).diagonal().array().sqrt();
    stateStdDev.imuLinearVelocity =  P.block<3, 3>(imuLinearVelOffset, imuLinearVelOffset).diagonal().array().sqrt();

//     stateStdDev.lContactFramePosition =  P.block<3, 3>(m_vecOffsets.lContactFramePosition, m_vecOffsets.lContactFramePosition).diagonal().array().sqrt();
//     stateStdDev.lContactFrameOrientation =  P.block<3, 3>(m_vecOffsets.lContactFrameOrientation, m_vecOffsets.lContactFrameOrientation).diagonal().array().sqrt();
// 
//     stateStdDev.rContactFramePosition =  P.block<3, 3>(m_vecOffsets.rContactFramePosition, m_vecOffsets.rContactFramePosition).diagonal().array().sqrt();
//     stateStdDev.rContactFrameOrientation =  P.block<3, 3>(m_vecOffsets.rContactFrameOrientation, m_vecOffsets.rContactFrameOrientation).diagonal().array().sqrt();
 
    if (estimateBias)
    {
        auto accBiasOffset = P.cols() - accBiasOffsetFromTail;
        auto gyroBiasOffset = P.cols() - gyroBiasOffsetFromTail;
        stateStdDev.accelerometerBias =  P.block<3, 3>(accBiasOffset, accBiasOffset).diagonal().array().sqrt();
        stateStdDev.gyroscopeBias =  P.block<3, 3>(gyroBiasOffset, gyroBiasOffset).diagonal().array().sqrt();        
    }
}

void DILIGENT::Impl::constuctStateVar(const FloatingBaseEstimators::StateStdDev& stateStdDev,
                                      const bool& estimateBias,
                                      Eigen::MatrixXd& P)
{
//     if (estimateBias)
//     {
//         P.resize(m_vecSizeWBias, m_vecSizeWBias);
//     }
//     else
//     {
//         P.resize(m_vecSizeWOBias, m_vecSizeWOBias);
//     }
// 
//     P.setZero();
//     Eigen::Vector3d temp;
// 
//     temp = stateStdDev.imuPosition.array().square();
//     P.block<3, 3>(m_vecOffsets.imuPosition, m_vecOffsets.imuPosition) = temp.asDiagonal();
//     temp = stateStdDev.imuOrientation.array().square();
//     P.block<3, 3>(m_vecOffsets.imuOrientation, m_vecOffsets.imuOrientation) = temp.asDiagonal();
//     temp = stateStdDev.imuLinearVelocity.array().square();
//     P.block<3, 3>(m_vecOffsets.imuLinearVel, m_vecOffsets.imuLinearVel) = temp.asDiagonal();
// 
//     temp = stateStdDev.lContactFramePosition.array().square();
//     P.block<3, 3>(m_vecOffsets.lContactFramePosition, m_vecOffsets.lContactFramePosition) = temp.asDiagonal();
//     temp = stateStdDev.lContactFrameOrientation.array().square();
//     P.block<3, 3>(m_vecOffsets.lContactFrameOrientation, m_vecOffsets.lContactFrameOrientation) = temp.asDiagonal();
// 
//     temp = stateStdDev.rContactFramePosition.array().square();
//     P.block<3, 3>(m_vecOffsets.rContactFramePosition, m_vecOffsets.rContactFramePosition) = temp.asDiagonal();
//     temp = stateStdDev.rContactFrameOrientation.array().square();
//     P.block<3, 3>(m_vecOffsets.rContactFrameOrientation, m_vecOffsets.rContactFrameOrientation) = temp.asDiagonal();
// 
//     if (estimateBias)
//     {
//         temp = stateStdDev.gyroscopeBias.array().square();
//         P.block<3, 3>(m_vecOffsets.gyroBias, m_vecOffsets.gyroBias) = temp.asDiagonal();
//         temp = stateStdDev.accelerometerBias.array().square();
//         P.block<3, 3>(m_vecOffsets.accBias, m_vecOffsets.accBias) = temp.asDiagonal();
//     }
}

void DILIGENT::Impl::constructStateLieGroup(const FloatingBaseEstimators::InternalState& state,
                                            const bool& estimateBias,
                                            FloatingBaseExtendedKinematicsLieGroup& G)
{
    manif::SO3d Rb(state.imuOrientation);
    auto Xb = manif::SE_2_3d(state.imuPosition, Rb, state.imuLinearVelocity);
    G.setBaseExtendedPose(Xb);
    
    G.clearSupportFrames();
    for (auto& [idx, supportFrame] : state.supportFrameData)
    {
        G.addSupportFramePose(idx, supportFrame.pose);
    }

    // offset ids for landmarks - expected landmarkData map to maintain only positive ids
    for (auto& [idx, landmark] : state.landmarkData)
    {
        G.addSupportFramePose(landmarkIDsOffset + idx, landmark.pose);
    }    

    if (estimateBias)
    {
        Eigen::VectorXd imuBias(6);
        imuBias << state.accelerometerBias, state.gyroscopeBias;
        G.setAugmentedVector(imuBias);
    }
}

bool DILIGENT::Impl::extractStateFromLieGroup(const FloatingBaseExtendedKinematicsLieGroup& G,
                                              const bool& estimateBias,
                                              FloatingBaseEstimators::InternalState& state)
{
    state.imuPosition = G.basePosition();
    state.imuOrientation = G.baseRotation().quat();
    state.imuLinearVelocity = G.baseLinearVelocity();

    auto poses = G.supportFramesPose();
    for (auto& [idx, pose] : poses)
    {
        if (idx < landmarkIDsOffset)
        {
            if (state.supportFrameData.find(idx) == state.supportFrameData.end())
            {
                std::cerr << "[DILIGENT::extractStateFromLieGroup] handling a non-existent support frame." << std::endl;
                return false;
            }
            
            state.supportFrameData.at(idx).pose = pose;
        }
        else
        {
            auto ldmkIdx = idx - landmarkIDsOffset;
            if (state.landmarkData.find(ldmkIdx) == state.landmarkData.end())
            {
                std::cerr << "[DILIGENT::extractStateFromLieGroup] handling a non-existent landmark." << std::endl;
                return false;
            }
            
            state.landmarkData.at(ldmkIdx).pose = pose;
        }
    }

    if (G.isAugmentedVectorUsed() != estimateBias)
    {
        std::cerr << "[DILIGENT::extractStateFromLieGroup] bias handling mismatch" << std::endl;
        return false;
    }
    else
    {
        if (G.isAugmentedVectorUsed())
        {
            Eigen::VectorXd imuBias;
            G.augmentedVector(imuBias);
            if (imuBias.size() != 6)
            {
                std::cerr << "[DILIGENT::extractStateFromLieGroup] bias size mismatch" << std::endl;
                return false;
            }
            state.accelerometerBias = imuBias.head<3>();
            state.gyroscopeBias = imuBias.tail<3>();
        }
    }
    
    return true;
}

void DILIGENT::Impl::calcSlantFk(const FloatingBaseEstimators::InternalState& state,
                                 const size_t dim,
                                 const double& dt,
                                 const Eigen::Vector3d& g,
                                 const bool& estimateBias,
                                 Eigen::MatrixXd& slantFk)
{
    // When biases are enabled,
    // slantFk = [   0  [a]x   I_3 dt   ...   -I_3 (dt^2) 0.5        0]
    //           [   0     0        0   ...                 0  -I_3 dt]
    //           [   0  [g]x        0   ...           -I_3 dt        0]
    //           [   .     .        .   ...                 .        .]
    //           [   .     .        .   ...                 .        .]
    //           [   .     .        .   ...                 .        .]
    //           [   0     0        0   ...                 0        0]
    //           [   0     0        0   ...                 0        0]
    // when biases are disabled, ignore last two rows and columns
    // the dots are filled with zeros depending on the number of support frame and landmarks
        
    slantFk.resize(dim, dim);
    slantFk.setZero();
 
    Eigen::Matrix3d RT = state.imuOrientation.toRotationMatrix();
    const Eigen::Vector3d& v = state.imuLinearVelocity;

    Eigen::Vector3d a = RT*v*dt + RT*g*0.5*dt*dt;
    Eigen::Matrix3d aCross = iDynTree::skew(a);
    Eigen::Matrix3d gCross = iDynTree::skew(RT*g*dt);

    slantFk.block<3, 3>(imuPositionOffset, imuOrientationOffset) = aCross;
    slantFk.block<3, 3>(imuPositionOffset, imuLinearVelOffset) = I3*dt;
    slantFk.block<3, 3>(imuLinearVelOffset, imuOrientationOffset) = gCross;

    if (estimateBias)
    {
        auto accBiasOffset = dim - accBiasOffsetFromTail;
        auto gyroBiasOffset = dim - gyroBiasOffsetFromTail;
        slantFk.block<3, 3>(imuPositionOffset, accBiasOffset) = -I3*dt*dt*0.5;
        slantFk.block<3, 3>(imuOrientationOffset, gyroBiasOffset) = -I3*dt;
        slantFk.block<3, 3>(imuLinearVelOffset, accBiasOffset) = -I3*dt;
    }
}

void DILIGENT::Impl::calcQk(const FloatingBaseEstimators::SensorsStdDev& sensStdDev,
                            const FloatingBaseEstimators::InternalState& state,
                            const FloatingBaseExtendedKinematicsLieGroupTangent& Omegak,
                            const size_t dim,
                            const double& dt,
                            const bool& estimateBias,
                            Eigen::MatrixXd& Qk)
{
    // When biases are enabled,
    // Qc = blkdiag(0.5 Qa dt^2, Qg dt, Qa dt, {Qf dt}, {Ql dt}, Qbg dt, Qba dt)
    // Qf the support frame noise params depend on support frame contact states
    Qk.resize(dim, dim);
    Qk.setZero();

    Eigen::Vector3d Qa = sensStdDev.accelerometerNoise.array().square();
    Eigen::Vector3d Qg = sensStdDev.gyroscopeNoise.array().square();

    Qk.block<3, 3>(imuPositionOffset, imuPositionOffset) = 0.5*dt*dt*static_cast<Eigen::Matrix3d>(Qa.asDiagonal());
    Qk.block<3, 3>(imuOrientationOffset, imuOrientationOffset) = dt*static_cast<Eigen::Matrix3d>(Qg.asDiagonal());
    Qk.block<3, 3>(imuLinearVelOffset, imuLinearVelOffset) = dt*static_cast<Eigen::Matrix3d>(Qa.asDiagonal());
    
    Eigen::Matrix<double, 6, 1> Qf;
    Eigen::Matrix<double, 6, 1> supportFrameNoise;
    
    // get sorted list of indices
    auto supportFrameIds = Omegak.supportFrameIndices();
    
    int idx = 0;
    for (auto& frameIdx : supportFrameIds)
    {
        // prepare the noise covariance submatrix
        if (frameIdx < landmarkIDsOffset)
        {
            if (state.supportFrameData.at(frameIdx).isActive)
            {
                supportFrameNoise << sensStdDev.contactFootLinvelNoise, sensStdDev.contactFootAngvelNoise;
            }
            else
            {
                supportFrameNoise << sensStdDev.swingFootLinvelNoise, sensStdDev.swingFootAngvelNoise;
            }
        }
        else
        {
            supportFrameNoise << sensStdDev.landmarkPredictionNoise;
        }
        
        Qf = supportFrameNoise.array().square();
        
        int q = extMotionDim + (motionDim*idx);
        Qk.block<6, 6>(q, q) = dt*static_cast<Eigen::Matrix<double, 6, 6> >(Qf.asDiagonal());
        idx++;
    }
    
    if (estimateBias)
    {
        auto accBiasOffset = dim - accBiasOffsetFromTail;
        auto gyroBiasOffset = dim - gyroBiasOffsetFromTail;
        Eigen::Vector3d Qba = sensStdDev.accelerometerBiasNoise.array().square();
        Eigen::Vector3d Qbg = sensStdDev.gyroscopeBiasNoise.array().square();
        Qk.block<3, 3>(accBiasOffset, accBiasOffset) = dt*static_cast<Eigen::Matrix3d>(Qba.asDiagonal());
        Qk.block<3, 3>(gyroBiasOffset, gyroBiasOffset) = dt*static_cast<Eigen::Matrix3d>(Qbg.asDiagonal());        
    }
}

void DILIGENT::Impl::calcOmegak(const FloatingBaseEstimators::InternalState& X,
                                const FloatingBaseEstimators::Measurements& meas,
                                const double& dt,
                                const Eigen::Vector3d& g,
                                const bool& estimateBias,
                                FloatingBaseExtendedKinematicsLieGroupTangent& Omegak)
{
    // F = nr of contact frames
    // D = nr of landmarks
    // When biases are enabled,
    // _()_ = [ R.T v dt + 0.5 a dt^2 ]
    //        [        w dt           ]
    //        [        a dt           ]
    //        [      0_{6Fx1}         ]
    //        [      0_{6Dx1}         ]
    //        [       0_{6x1}         ]
    //
    // when biases are disabled, last six rows are omitted
    // w = y_gyro - b_gyro
    // a = y_acc - b_acc + R.T g
    
    Eigen::Matrix3d RT = X.imuOrientation.toRotationMatrix().transpose();
    Eigen::Vector3d w = meas.gyro - X.gyroscopeBias;
    Eigen::Vector3d a = meas.acc - X.accelerometerBias + (RT * g);
    const Eigen::Vector3d& v = X.imuLinearVelocity;
    
    Eigen::VectorXd vBase(9);
    vBase << (RT*v*dt) + (a*0.5*dt*dt), w*dt, a*dt;
    auto baseMotion = manif::SE_2_3Tangentd(vBase);
    Omegak.setBaseExtendedMotionVector(baseMotion);

    std::map<int, manif::SE3Tangentd> supportFrameMap;    

    Omegak.clearSupportFrames();
    // positive ids for contacts
    for (auto& iter : X.supportFrameData)
    {
        Omegak.addSupportFrameTwist(iter.first, zeroTwist);
    }

    // offset ids for landmarks - expected landmarkData map to maintain only positive ids
    for (auto& iter : X.landmarkData)
    {
        Omegak.addSupportFrameTwist(landmarkIDsOffset + iter.first, zeroTwist);
    }    

    if (estimateBias)
    {
        Eigen::VectorXd vBias = Eigen::VectorXd::Zero(6);
        Omegak.setAugmentedVector(vBias);
    }
}
