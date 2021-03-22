/**
 * @file DILIGENT.cpp
 * @authors Prashanth Ramadoss
 * @copyright 2020 Istituto Italiano di Tecnologia (IIT). This software may be modified and
 * distributed under the terms of the GNU Lesser General Public License v2.1 or any later version.
 */

#include <BipedalLocomotion/FloatingBaseEstimators/DILIGENT.h>
#include <BipedalLocomotion/FloatingBaseEstimators/FloatingBaseExtendedKinematicsLieGroup.h>
#include <BipedalLocomotion/Conversions/ManifConversions.h>

#include <manif/manif.h>

using namespace BipedalLocomotion::Estimators;
using namespace BipedalLocomotion::Conversions;

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
    bool constuctStateVar(const FloatingBaseEstimators::InternalState& state,
                          const FloatingBaseEstimators::StateStdDev& stateStdDev,
                          const bool& estimateBias, Eigen::MatrixXd& P);

    /**
     * Extract internal state standard deviation object,
     * from the diagonal elements of the state covariance matrix
     */
    bool extractStateVar(const Eigen::MatrixXd& P,
                         const FloatingBaseExtendedKinematicsLieGroup& Xk,
                         const bool& estimateBias,
                         FloatingBaseEstimators::StateStdDev& stateStdDev);

    /**
     * Compute left parametrized velocity
     */
    void calcOmegak(const FloatingBaseEstimators::InternalState& X,
                    const FloatingBaseEstimators::Measurements& meas,
                    const double& dt, const Eigen::Vector3d& g,
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
    bool updateStates(const Eigen::VectorXd& deltaY, const Eigen::MatrixXd H,
                      const Eigen::MatrixXd& N, const bool& estimateBias,
                      FloatingBaseExtendedKinematicsLieGroup& Xk,
                      FloatingBaseEstimators::InternalState& state,
                      Eigen::MatrixXd& P);


    /**
     * Compute continuous time system noise covariance matrix
     * using the predicted internal state estimates
     */
    void calcQk(const FloatingBaseEstimators::SensorsStdDev& sensStdDev,
                const FloatingBaseEstimators::InternalState& state,
                const FloatingBaseExtendedKinematicsLieGroupTangent& Omegak,
                const size_t dim, const double& dt,
                const bool& estimateBias, Eigen::MatrixXd& Qc);

    /**
     * Compute left parametrized velocity Jacobian
     */
    void calcSlantFk(const FloatingBaseEstimators::InternalState& X,
                     const size_t dim, const double& dt,
                     const Eigen::Vector3d& g, const bool& estimateBias,
                     Eigen::MatrixXd& slantFk);

    bool vec2Tangent(const Eigen::VectorXd& v,
                     const std::vector<int>& ids, const bool& estimateBias,
                     FloatingBaseExtendedKinematicsLieGroupTangent& tangent);

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

    template <typename T, typename V>
    bool compareKeys(const std::map<int, T>& lhs, const std::map<int, V>& rhs)
    {
        return (lhs.size() == rhs.size()) &&
               (std::equal(lhs.begin(), lhs.end(), rhs.begin(),
                           [] (auto a, auto b) { return a.first == b.first; }));
    }


    bool addContactOrLandmark(const int& idx,  const double& time, const bool& isActive,
                              const manif::SE3d& poseEstimate,
                              const FloatingBaseEstimators::SensorsStdDev& sensStdDev,
                              const double& dt, const bool& estimateBias,
                              FloatingBaseEstimators::InternalState& state,
                              Eigen::MatrixXd& P);

    bool removeContactOrLandmark(const int& idx, const bool& estimateBias,
                                 FloatingBaseEstimators::InternalState& state,
                                 Eigen::MatrixXd& P);

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

    std::vector<int> existingContact;
    std::vector<int> existingLdmk;

    /** The following buffers are updated automatically through m_state and m_P*/
    FloatingBaseExtendedKinematicsLieGroup Xk;
    FloatingBaseExtendedKinematicsLieGroupTangent Omegak;
    FloatingBaseExtendedKinematicsLieGroupTangent negOmegak;
    Eigen::MatrixXd slantFk, Qk;
    Eigen::MatrixXd AdjNegativeOmegaLifted;
    Eigen::MatrixXd Jr; /**<right Jacobian of Lie group at Omega as tangent */
    Eigen::MatrixXd Fk; /**< system dynamics Jacobian*/
    Eigen::MatrixXd PHT;
    Eigen::MatrixXd H; /**< Measurement model Jacobian */
    Eigen::MatrixXd N; /**< Measurement noise covariance matrix */
    Eigen::MatrixXd S; /**< innovation covariance*/
    Eigen::MatrixXd Sinv; /**< inverse of innovation covariance*/
    Eigen::MatrixXd K; /**< Kalman gain*/
    Eigen::VectorXd deltaY; /**< innovation in vector space*/
    Eigen::VectorXd deltaX; /**< state update in vector space*/
    FloatingBaseExtendedKinematicsLieGroupTangent dX;
    Eigen::MatrixXd dXJr; /**<right Jacobian of Lie group at state correction as tangent */
    Eigen::MatrixXd IminusKH;
    Eigen::MatrixXd Fnew; /**< required for adding or removing contacts */
    Eigen::MatrixXd Qnew; /**< required for adding or removing contacts */
    Eigen::Matrix<double, 6, 1> Qf; /**< required for adding or removing contacts */
    Eigen::Matrix<double, 6, 1> supportFrameNoise; /**< required for adding or removing contacts */
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

    // construct priors
    if (!m_pimpl->constuctStateVar(m_state, m_priors, m_options.imuBiasEstimationEnabled, m_pimpl->m_P))
    {
        std::cerr << "[DILIGENT::customInitialization] Could not construct initial state covariance matrix."
        << std::endl;
        return false;
    }

    return true;
}

bool DILIGENT::resetEstimator(const FloatingBaseEstimators::InternalState& newState,
                              const FloatingBaseEstimators::StateStdDev& newPriorDev)
{
    m_state = newState;
    m_stateStdDev = newPriorDev;
    m_priors = newPriorDev;

    if (!m_pimpl->constuctStateVar(m_state, m_priors, m_options.imuBiasEstimationEnabled, m_pimpl->m_P))
    {
        std::cerr << "[DILIGENT::resetEstimator] Could not construct state covariance matrix."
        << std::endl;
        return false;
    }

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
        std::cerr << "[DILIGENT::predictState] unable to propagate state mean." << std::endl;
        return false;
    }

    if (m_pimpl->Fk.cols() != m_pimpl->m_P.rows())
    {
        std::cerr << "[DILIGENT::predictState] covariance matrices size mismatch" << std::endl;
        return false;
    }

    m_pimpl->m_P = m_pimpl->Fk*m_pimpl->m_P*(m_pimpl->Fk.transpose()) +
                   m_pimpl->Jr*(m_pimpl->Qk*dt)*(m_pimpl->Jr.transpose());

    if (!m_pimpl->extractStateVar(m_pimpl->m_P,
                                  m_pimpl->Xk,
                                  m_options.imuBiasEstimationEnabled,
                                  m_stateStdDev))// unwrap state covariance matrix diagonal
    {
        std::cerr << "[DILIGENT::predictState] unable to extract from predicted state covariance matrix." << std::endl;
        return false;
    }

    return true;
}

bool DILIGENT::updateKinematics(FloatingBaseEstimators::Measurements& meas,
                                const double& dt)
{
    if (meas.stampedContactsStatus.size() < 1)
    {
        // early return, nothing to do if no measurements available
        return false;
    }

    // extract current predicted state
    Eigen::Matrix3d R = m_state.imuOrientation.toRotationMatrix();
    const Eigen::Vector3d& p = m_state.imuPosition;
    manif::SE3d imuPosePred = Conversions::toManifPose(R, p);

    Eigen::VectorXd encodersVar = m_sensorsDev.encodersNoise.array().square();
    Eigen::MatrixXd Renc = static_cast<Eigen::MatrixXd>(encodersVar.asDiagonal());

    // for each measurement
    // check for existing contact
    // if not available add new contact
    m_pimpl->existingContact.clear();
    for (auto& iter : m_state.supportFrameData)
    {
        m_pimpl->existingContact.push_back(iter.first);
    }

    std::unordered_map<int, manif::SE3d> measPoses;
    size_t nrContacts{0};
    // first iterate through measurement and add new contacts
    for (auto& iter : meas.stampedContactsStatus)
    {
        auto contactid = iter.first;
        auto relContactPose = Conversions::toManifPose(m_modelComp.kinDyn()->getRelativeTransform(m_modelComp.baseIMUIdx(), contactid));
        measPoses[contactid] = relContactPose;
        auto contactData = iter.second;
        bool isActive{contactData.isActive};
        auto timestamp = contactData.lastUpdateTime;

        if (std::find(m_pimpl->existingContact.begin(), m_pimpl->existingContact.end(),
            iter.first) == m_pimpl->existingContact.end())
        {
            if (!m_pimpl->addContactOrLandmark(contactid, timestamp, isActive,
                                               imuPosePred*relContactPose,
                                               m_sensorsDev, m_dt,
                                               m_options.imuBiasEstimationEnabled,
                                               m_state, m_pimpl->m_P))
            {
                continue;
            }
        }
        else
        {
            m_state.supportFrameData.at(contactid).isActive = isActive;
            m_state.supportFrameData.at(contactid).lastUpdateTime = timestamp;
        }

        if (isActive)
        {
            nrContacts++;
        }
    }

    // update the underlying matrix Lie group from the state
    m_pimpl->constructStateLieGroup(m_state, m_options.imuBiasEstimationEnabled, m_pimpl->Xk);

    // construct stack sub H and sub deltaY
    // construct blkdiag of sub N
    // When a contact is made at F, the measurement model is as follows
    //
    // h(X) = [A_R_B.T A_R_F  A_R_B.T (pf - p) ]   \in SE(3)
    //        [                             1  ]
    // The innovation term is in the Lie algebra of the SE(3),
    // deltaY = logvee_SE3(hinv, y)
    //
    // The measurement model Jacobian for two contacts RF and LF becomes, if biases are included,
    // H = [-A_R_LF.T A_R_IMU | -A_R_LF.T [p - plf]x A_R_IMU |  0_3 |   I | 0_3 | 0_3 | 0_3 | 0_3 | 0_3 ]
    //     [              0_3 |            -A_R_LF.T A_R_IMU |  0_3 | 0_3 |   I | 0_3 | 0_3 | 0_3 | 0_3 ]
    //     [-A_R_RF.T A_R_IMU | -A_R_RF.T [p - plf]x A_R_IMU |  0_3 | 0_3 | 0_3 |   I | 0_3 | 0_3 | 0_3 ]
    //     [              0_3 |            -A_R_RF.T A_R_IMU |  0_3 | 0_3 | 0_3 | 0_3 |   I | 0_3 | 0_3 ]
    // I is the 3d identity matrix, []x is the 3d skew symmetric matrix
    // If biases are enabled, the last 6 columns are not considered
    //
    // The measurement noise covariance sub-block corrsponding to F,
    // Rc = F_J_{IMU,LF} Renc LF_J_{IMU,LF}.T
    // N = Rc/dt is the discretized measurement noise covariance matrix
    // The measurement noise is left-trivialized forward kinematic velocity noise which is computed
    // using the manipulator Jacobian of the foot with respect to the IMU
    //
    // For multiple contacts, the H matrices are stacked and N matrices are block diagonalized
    const int stateSpaceDims = m_pimpl->dimensions(m_state, m_options.imuBiasEstimationEnabled);
    const int measurementSpaceDims = nrContacts*m_pimpl->motionDim;
    m_pimpl->H.resize(measurementSpaceDims, stateSpaceDims);
    m_pimpl->H.setZero();
    m_pimpl->N.resize(measurementSpaceDims, measurementSpaceDims);
    m_pimpl->N.setZero();
    m_pimpl->deltaY.resize(measurementSpaceDims);
    m_pimpl->deltaY.setZero();

    int k = 0; // measurement space index jump
    for (auto& obs : meas.stampedContactsStatus)
    {
        int contactOffsetMeas = k*m_pimpl->motionDim;
        int j = 0; // state space index jump
        for (auto& iter : m_state.supportFrameData)
        {
            auto& contactId = obs.first;
            auto& contactState = obs.second.isActive;
            if (iter.first == contactId && contactState) // update if contact is active
            {
                int contactOffsetState = m_pimpl->extMotionDim  + (j*m_pimpl->motionDim);
                Eigen::Matrix3d Z = m_state.supportFrameData.at(iter.first).pose.asSO3().rotation();
                Eigen::Vector3d d = m_state.supportFrameData.at(iter.first).pose.translation();

                auto hOfX = Conversions::toManifPose(R.transpose()*Z, R.transpose()*(d - p));
                auto& y = measPoses.at(contactId);

                // innovation
                auto poseError = y - hOfX;  // this performs logvee_SE3(inv(hOfX), y)
                m_pimpl->deltaY.segment<6>(contactOffsetMeas) = poseError.coeffs();

                // measurement noise covariance
                Eigen::MatrixXd F_J_IMUF(m_pimpl->motionDim, m_modelComp.kinDyn()->getNrOfDegreesOfFreedom());
                bool ok = m_modelComp.kinDyn()->getRelativeJacobianExplicit(m_modelComp.baseIMUIdx(), contactId,
                                                                           contactId, contactId, F_J_IMUF);
                m_pimpl->N.block<6, 6>(contactOffsetMeas, contactOffsetMeas) = F_J_IMUF*Renc*(F_J_IMUF.transpose());

                // measurement model Jacobian
                Eigen::MatrixXd jacIMU(m_pimpl->motionDim, m_pimpl->motionDim);
                // [  -Z.T R  | -Z.T [p - d]x R  | 0_3 ... I_3 | 0_3 ...]
                // [     0_3  |          -Z.T R  | 0_3 ... 0_3 | I_3 ...]
                jacIMU <<  -Z.transpose()*R, -Z.transpose()*manif::skew(p - d)*R,
                    Eigen::Matrix3d::Zero(), -Z.transpose()*R;

                m_pimpl->H.block<6, 6>(contactOffsetMeas, m_pimpl->imuPositionOffset) = jacIMU;
                m_pimpl->H.block<6, 6>(contactOffsetMeas, contactOffsetState).setIdentity();

                // only if active contact
                // increase k count
                k++;
                break;
            }
            j++;
        }
    }

    // update state and covariance and clear used measurements
    if (nrContacts > 0)
    {
        // discretize the measurement noise covariance
        m_pimpl->N /= dt;

        meas.stampedContactsStatus.clear();
        if (!m_pimpl->updateStates(m_pimpl->deltaY, m_pimpl->H, m_pimpl->N,
                                   m_options.imuBiasEstimationEnabled, m_pimpl->Xk,
                                   m_state, m_pimpl->m_P))
        {
            std::cerr << "[DILIGENT::updateKinematics] unable to correct states with landmark measurements." << std::endl;
            return false;
        }

        if (!m_pimpl->extractStateVar(m_pimpl->m_P,
                                      m_pimpl->Xk,
                                      m_options.imuBiasEstimationEnabled,
                                      m_stateStdDev))// unwrap state covariance matrix diagonal
        {
            std::cerr << "[DILIGENT::updateKinematics] unable to extract from predicted state covariance matrix." << std::endl;
            return false;
        }
    }

    return true;
}

bool DILIGENT::updateLandmarkRelativePoses(FloatingBaseEstimators::Measurements& meas,
                                           const double& dt)
{
    Eigen::Matrix3d R = m_state.imuOrientation.toRotationMatrix();
    const Eigen::Vector3d& p = m_state.imuPosition;
    manif::SE3d imuPosePred = Conversions::toManifPose(R, p);

    // for each measurement
    // check for existing landmark
    // if not available add new landmark
    m_pimpl->existingLdmk.clear();
    for (auto& iter : m_state.landmarkData)
    {
        m_pimpl->existingLdmk.push_back(iter.first);
    }

    // first iterate through measurement and add new landmarks
    for (auto& iter : meas.stampedRelLandmarkPoses)
    {
        if (std::find(m_pimpl->existingLdmk.begin(), m_pimpl->existingLdmk.end(),
            iter.first) == m_pimpl->existingLdmk.end())
        {
            auto ldmkid = m_pimpl->landmarkIDsOffset+iter.first;
            auto ldmkData = iter.second;
            auto timestamp = ldmkData.lastUpdateTime;
            auto relLdmkPose = ldmkData.pose;
            bool isActive{true};

            if (!m_pimpl->addContactOrLandmark(ldmkid, timestamp, isActive,
                                               imuPosePred*relLdmkPose,
                                               m_sensorsDev, m_dt,
                                               m_options.imuBiasEstimationEnabled,
                                               m_state, m_pimpl->m_P))
            {
                continue;
            }
        }
    }

    // update the underlying matrix Lie group from the state
    m_pimpl->constructStateLieGroup(m_state, m_options.imuBiasEstimationEnabled, m_pimpl->Xk);

    // construct stack sub H and sub deltaY
    // construct blkdiag of sub N
    const int stateSpaceDims = m_pimpl->dimensions(m_state, m_options.imuBiasEstimationEnabled);
    const int measurementSpaceDims = meas.stampedRelLandmarkPoses.size()*m_pimpl->motionDim;
    m_pimpl->H.resize(measurementSpaceDims, stateSpaceDims);
    m_pimpl->H.setZero();
    m_pimpl->N.resize(measurementSpaceDims, measurementSpaceDims);
    m_pimpl->N.setZero();
    m_pimpl->deltaY.resize(measurementSpaceDims);
    m_pimpl->deltaY.setZero();

    int k = 0; // measurement space index jump
    for (auto& obs : meas.stampedRelLandmarkPoses)
    {
        int ldmkOffsetMeas = k*m_pimpl->motionDim;
        int j = m_state.supportFrameData.size(); // state space index jump
        for (auto& iter : m_state.landmarkData)
        {
            if (iter.first == obs.first)
            {
                int ldmkOffsetState = m_pimpl->extMotionDim  + (j*m_pimpl->motionDim);
                Eigen::Matrix3d Z = m_state.landmarkData.at(iter.first).pose.asSO3().rotation();
                Eigen::Vector3d d = m_state.landmarkData.at(iter.first).pose.translation();

                auto hOfX = Conversions::toManifPose(R.transpose()*Z, R.transpose()*(d - p));
                auto& y = obs.second.pose;

                // innovation
                auto poseError = y - hOfX;  // this performs logvee_SE3(inv(hOfX), y)
                m_pimpl->deltaY.segment<6>(ldmkOffsetMeas) = poseError.coeffs();

                // measurement noise covariance
                Eigen::VectorXd ldmkVar = m_sensorsDev.landmarkMeasurementNoise.array().square();
                m_pimpl->N.block<6, 6>(ldmkOffsetMeas, ldmkOffsetMeas) = static_cast<Eigen::MatrixXd>(ldmkVar.asDiagonal());

                // measurement model Jacobian
                Eigen::MatrixXd jacIMU(m_pimpl->motionDim, m_pimpl->motionDim);
                // [  -Z.T R  | -Z.T [p - d]x R  | 0_3 ... I_3 | 0_3 ...]
                // [     0_3  |          -Z.T R  | 0_3 ... 0_3 | I_3 ...]
                jacIMU <<  -Z.transpose()*R, -Z.transpose()*manif::skew(p - d)*R,
                    Eigen::Matrix3d::Zero(), -Z.transpose()*R;

                m_pimpl->H.block<6, 6>(ldmkOffsetMeas, m_pimpl->imuPositionOffset) = jacIMU;
                m_pimpl->H.block<6, 6>(ldmkOffsetMeas, ldmkOffsetState).setIdentity();
                break;
            }
            j++;
        }
        k++;
    }

    // update state and covariance and clear used measurements
    if (meas.stampedRelLandmarkPoses.size() > 0)
    {
        // discretize the measurement noise covariance
        m_pimpl->N /= dt;

        meas.stampedRelLandmarkPoses.clear();
        if (!m_pimpl->updateStates(m_pimpl->deltaY, m_pimpl->H, m_pimpl->N,
                                   m_options.imuBiasEstimationEnabled, m_pimpl->Xk,
                                   m_state, m_pimpl->m_P))
        {
            std::cerr << "[DILIGENT::updateLandmarkRelativePoses] unable to correct states with landmark measurements." << std::endl;
            return false;
        }

        if (!m_pimpl->extractStateVar(m_pimpl->m_P,
                                      m_pimpl->Xk,
                                      m_options.imuBiasEstimationEnabled,
                                      m_stateStdDev))// unwrap state covariance matrix diagonal
        {
            std::cerr << "[DILIGENT::updateLandmarkRelativePoses] unable to extract from predicted state covariance matrix." << std::endl;
            return false;
        }
    }

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
                                  const Eigen::MatrixXd Hk,
                                  const Eigen::MatrixXd& Nk,
                                  const bool& estimateBias,
                                  FloatingBaseExtendedKinematicsLieGroup& Xk,
                                  FloatingBaseEstimators::InternalState& state,
                                  Eigen::MatrixXd& P)
{
    if (Hk.cols() != P.rows())
    {
        std::cerr << "[DILIGENT::updateStates] Measurement model Jacobian size mismatch" << std::endl;
        return false;
    }

    if (Hk.rows() != Nk.rows())
    {
        std::cerr << "[DILIGENT::updateStates] Measurement noise covariance matrix size mismatch" << std::endl;
        return false;
    }

    if (Xk.dimensions() != P.rows())
    {
        std::cerr << "[DILIGENT::updateStates] underlying Lie group dimensions mismatch" << std::endl;
        return false;
    }

    PHT = P*Hk.transpose();
    S = Hk*PHT + Nk;
    Sinv = S.inverse();
    K = PHT*(Sinv);
    deltaX = K*deltaY;

    if (!vec2Tangent(deltaX, Xk.supportFrameIndices(), estimateBias, dX))
    {
        std::cerr << "[DILIGENT::updateStates] could not retrieve Lie group tangent" << std::endl;
        return false;
    }

    // update estimate
    if (!propagateStates(Xk, dX, estimateBias, state))
    {
        std::cerr << "[DILIGENT::updateStates] could not correct state estimate using measurement updates" << std::endl;
        return false;
    }

    // update covariance
    dXJr = dX.rjac();
    IminusKH = Eigen::MatrixXd::Identity(P.rows(), P.cols()) - K*Hk;
    P = dXJr*IminusKH*P*(dXJr.transpose());

    return true;
}

bool DILIGENT::Impl::extractStateVar(const Eigen::MatrixXd& P,
                                     const FloatingBaseExtendedKinematicsLieGroup& Xk,
                                     const bool& estimateBias,
                                     FloatingBaseEstimators::StateStdDev& stateStdDev)
{
    stateStdDev.imuPosition =  P.block<3, 3>(imuPositionOffset, imuPositionOffset).diagonal().array().sqrt();
    stateStdDev.imuOrientation =  P.block<3, 3>(imuOrientationOffset, imuOrientationOffset).diagonal().array().sqrt();
    stateStdDev.imuLinearVelocity =  P.block<3, 3>(imuLinearVelOffset, imuLinearVelOffset).diagonal().array().sqrt();

    std::size_t dimBias;
    estimateBias ? dimBias = 6 : dimBias = 0;
    auto dimSupport = P.cols() - dimBias - extMotionDim;

    auto supportFrameIds = Xk.supportFrameIndices();
    if (dimSupport != (motionDim*supportFrameIds.size()) )
    {
        std::cerr << "[DILIGENT::extractStateVar] support frame dimensions mismatch." << std::endl;
        return false;
    }

    int idx = 0;
    for (auto& frameIdx : supportFrameIds)
    {
        int q = extMotionDim + (motionDim*idx);
        if (frameIdx < landmarkIDsOffset)
        {
            stateStdDev.supportFramePose[idx] = P.block<6, 6>(q, q).diagonal().array().sqrt();
        }
        else
        {
            stateStdDev.landmarkPose[landmarkIDsOffset- idx] = P.block<6, 6>(q, q).diagonal().array().sqrt();
        }
        idx++;
    }

    if (estimateBias)
    {
        auto accBiasOffset = P.cols() - accBiasOffsetFromTail;
        auto gyroBiasOffset = P.cols() - gyroBiasOffsetFromTail;
        stateStdDev.accelerometerBias =  P.block<3, 3>(accBiasOffset, accBiasOffset).diagonal().array().sqrt();
        stateStdDev.gyroscopeBias =  P.block<3, 3>(gyroBiasOffset, gyroBiasOffset).diagonal().array().sqrt();
    }

    return true;
}

bool DILIGENT::Impl::constuctStateVar(const FloatingBaseEstimators::InternalState& state,
                                      const FloatingBaseEstimators::StateStdDev& stateStdDev,
                                      const bool& estimateBias,
                                      Eigen::MatrixXd& P)
{
    if (!compareKeys(state.supportFrameData, stateStdDev.supportFramePose))
    {
        std::cerr << "[DILIGENT::constuctStateVar] support frame data mismatch." << std::endl;
        return false;
    }

    if (!compareKeys(state.landmarkData, stateStdDev.landmarkPose))
    {
        std::cerr << "[DILIGENT::constuctStateVar] ladmark data mismatch." << std::endl;
        return false;
    }

    auto dim = dimensions(state, estimateBias);
    P.resize(dim, dim);
    P.setZero();
    Eigen::Vector3d temp;

    temp = stateStdDev.imuPosition.array().square();
    P.block<3, 3>(imuPositionOffset, imuPositionOffset) = temp.asDiagonal();
    temp = stateStdDev.imuOrientation.array().square();
    P.block<3, 3>(imuOrientationOffset, imuOrientationOffset) = temp.asDiagonal();
    temp = stateStdDev.imuLinearVelocity.array().square();
    P.block<3, 3>(imuLinearVelOffset, imuLinearVelOffset) = temp.asDiagonal();

    int idx = 0;
    Eigen::Matrix<double, 6, 1> frameVar;
    for (auto& [frameIdx, stddev] : stateStdDev.supportFramePose)
    {
        frameVar = stddev.array().square();
        int q = extMotionDim + (motionDim*idx);
        P.block<6, 6>(q, q) = frameVar.asDiagonal();
        idx++;
    }

    for (auto& [frameIdx, stddev] : stateStdDev.landmarkPose)
    {
        frameVar = stddev.array().square();
        int q = extMotionDim + (motionDim*idx);
        P.block<6, 6>(q, q) = frameVar.asDiagonal();
        idx++;
    }

    if (estimateBias)
    {
        Eigen::Matrix<double, 6, 1> biasVar;
        biasVar << stateStdDev.accelerometerBias.array().square(), stateStdDev.gyroscopeBias.array().square();
        P.bottomRightCorner<6, 6>() = biasVar.asDiagonal();
    }
    return true;
}

void DILIGENT::Impl::constructStateLieGroup(const FloatingBaseEstimators::InternalState& state,
                                            const bool& estimateBias,
                                            FloatingBaseExtendedKinematicsLieGroup& G)
{
    auto Hb = toManifPose(state.imuOrientation.toRotationMatrix(), state.imuPosition);
    auto Xb = manif::SE_2_3d(Hb.isometry(), state.imuLinearVelocity);
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

    Eigen::Matrix3d RT = state.imuOrientation.toRotationMatrix().transpose();
    const Eigen::Vector3d& v = state.imuLinearVelocity;

    Eigen::Vector3d a = RT*v*dt + RT*g*0.5*dt*dt;
    Eigen::Matrix3d aCross = manif::skew(a);
    Eigen::Matrix3d gCross = manif::skew(RT*g*dt);

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

    // positive ids for contacts
    for (auto& iter : X.supportFrameData)
    {
        if (!Omegak.frameExists(iter.first))
        {
            Omegak.addSupportFrameTwist(iter.first, zeroTwist);
        }
        else
        {
            Omegak.setSupportFrameTwist(iter.first, zeroTwist);
        }
    }

    // offset ids for landmarks - expected landmarkData map to maintain only positive ids
    for (auto& iter : X.landmarkData)
    {
        if (!Omegak.frameExists(landmarkIDsOffset + iter.first))
        {
            Omegak.addSupportFrameTwist(landmarkIDsOffset + iter.first, zeroTwist);
        }
        else
        {
            Omegak.setSupportFrameTwist(landmarkIDsOffset + iter.first, zeroTwist);
        }
    }

    if (estimateBias)
    {
        // technically not a zero twist
        // neverthelss an existing store of 6 zeros
        Omegak.setAugmentedVector(zeroTwist.coeffs());
    }
}


bool DILIGENT::Impl::vec2Tangent(const Eigen::VectorXd& v,
                                 const std::vector<int>& ids,
                                 const bool& estimateBias,
                                 FloatingBaseExtendedKinematicsLieGroupTangent& tangent)
{
    auto vecSize = v.size();
    if (vecSize < extMotionDim)
    {
        std::cerr << "[DILIGENT::vecToTangent] vector size must be atleast 9" << std::endl;
        return false;
    }

    std::size_t dimBias;
    estimateBias ? dimBias = 6 : dimBias = 0;
    auto dimSupport = vecSize - dimBias - extMotionDim;

    if ( ((vecSize - extMotionDim) % 6) != 0 || dimSupport / 6 != ids.size())
    {
        std::cerr << "[DILIGENT::vecToTangent] vector size mismatch" << std::endl;
        return false;
    }

    tangent.setBaseExtendedMotionVector(v.head(extMotionDim));

    int idx = 0;
    for (auto& id : ids)
    {
        int q = extMotionDim + (motionDim*idx);
        if (!tangent.frameExists(id))
        {
            tangent.addSupportFrameTwist(id, v.segment<6>(q));
        }
        else
        {
            tangent.setSupportFrameTwist(id, v.segment<6>(q));
        }
        idx++;
    }

    if (estimateBias)
    {
        Omegak.setAugmentedVector(v.tail(dimBias));
    }

    return true;
}

bool DILIGENT::Impl::addContactOrLandmark(const int& idx,
                                          const double& time,
                                          const bool& isActive,
                                          const manif::SE3d& poseEstimate,
                                          const FloatingBaseEstimators::SensorsStdDev& sensStdDev,
                                          const double& dt,
                                          const bool& estimateBias,
                                          FloatingBaseEstimators::InternalState& state,
                                          Eigen::MatrixXd& P)
{
    int oldCols = P.cols();
    int newRows = P.rows() + motionDim;
    int newCols = P.cols() + motionDim;
    Fnew.resize(newRows, oldCols);
    Fnew.setZero();
    Fnew.topLeftCorner(extMotionDim, extMotionDim).setIdentity();
    Qnew.resize(newRows, newCols);
    Qnew.setZero();
    if (idx < landmarkIDsOffset)
    {
        // add contact
        if (state.supportFrameData.find(idx) != state.supportFrameData.end())
        {
            std::cerr << "[DILIGENT::addContactOrLandmark] contact already exists" << std::endl;
            return false;
        }

        BipedalLocomotion::Contacts::EstimatedContact newContact;
        newContact.index = idx;
        newContact.name = "Contact#" + std::to_string(idx);
        newContact.lastUpdateTime = time;
        newContact.pose = poseEstimate;
        newContact.isActive = isActive;
        state.supportFrameData[idx] = newContact;

        if (isActive)
        {
            supportFrameNoise << sensStdDev.contactFootLinvelNoise, sensStdDev.contactFootAngvelNoise;
        }
        else
        {
            supportFrameNoise << sensStdDev.swingFootLinvelNoise, sensStdDev.swingFootAngvelNoise;
        }

        int j = 0;
        for (auto& iter : state.supportFrameData)
        {
            int q = extMotionDim + (j*motionDim);
            if (iter.first < idx)
            {
                Fnew.block<6, 6>(q, q).setIdentity();
            }
            else if (iter.first == idx)
            {
                Qf = supportFrameNoise.array().square();
                Qnew.block<6, 6>(q, q) = dt*static_cast<Eigen::Matrix<double, 6, 6> >(Qf.asDiagonal());
            }
            else if (iter.first > idx)
            {
                Fnew.block<6, 6>(q+motionDim, q).setIdentity();
            }
            j++;
        }

        for (auto& iter : state.landmarkData)
        {
            int q = extMotionDim  + (j*motionDim);
            Fnew.block<6, 6>(q+motionDim, q).setIdentity();
            j++;
        }
    }
    else
    {
        // add landmark
        int ldmkIdx = idx - landmarkIDsOffset;
        if (state.landmarkData.find(ldmkIdx) != state.landmarkData.end())
        {
            std::cerr << "[DILIGENT::addContactOrLandmark] landmark already exists" << std::endl;
            return false;
        }

        BipedalLocomotion::Contacts::EstimatedLandmark newLdmk;
        newLdmk.index = ldmkIdx;
        newLdmk.name = "Landmark#" + std::to_string(ldmkIdx);
        newLdmk.lastUpdateTime = time;
        newLdmk.pose = poseEstimate;
        newLdmk.isActive = true;
        state.landmarkData[ldmkIdx] = newLdmk;

        supportFrameNoise << sensStdDev.landmarkPredictionNoise;

        int j = 0;
        for (auto& iter : state.supportFrameData)
        {
            int q = extMotionDim  + (j*motionDim);
            Fnew.block<6, 6>(q, q).setIdentity();
            j++;
        }

        for (auto& iter : state.landmarkData)
        {
            int q = extMotionDim  + (j*motionDim);
            if (iter.first < ldmkIdx)
            {
                Fnew.block<6, 6>(q, q).setIdentity();
            }
            else if (iter.first == ldmkIdx)
            {
                Qf = supportFrameNoise.array().square();
                Qnew.block<6, 6>(q, q) = dt*static_cast<Eigen::Matrix<double, 6, 6> >(Qf.asDiagonal());
            }
            if (iter.first > ldmkIdx)
            {
                Fnew.block<6, 6>(q, q-motionDim).setIdentity();
            }
            j++;
        }
    }

    if (estimateBias)
    {
        Fnew.bottomRightCorner<6, 6>().setIdentity();
    }

    // update covariance
    P = Fnew * P * Fnew.transpose() + Qnew;

    return true;
}

bool DILIGENT::Impl::removeContactOrLandmark(const int& idx,
                                             const bool& estimateBias,
                                             FloatingBaseEstimators::InternalState& state,
                                             Eigen::MatrixXd& P)
{
    int oldCols = P.cols();
    int newRows = P.rows() - motionDim;

    Fnew.resize(newRows, oldCols);
    Fnew.setZero();
    Fnew.topLeftCorner(extMotionDim, extMotionDim).setIdentity();

    if (idx < landmarkIDsOffset)
    {
        // remove contact
        if (state.supportFrameData.find(idx) == state.supportFrameData.end())
        {
            std::cerr << "[DILIGENT::removeContactOrLandmark] contact does not exist" << std::endl;
            return false;
        }

        int j = 0;
        for (auto& iter : state.supportFrameData)
        {
            int q = extMotionDim + (j*motionDim);
            if (iter.first < idx)
            {
                Fnew.block<6, 6>(q, q).setIdentity();
            }
            else if (iter.first > idx)
            {
                Fnew.block<6, 6>(q-motionDim, q).setIdentity();
            }
            j++;
        }

        for (auto& iter : state.landmarkData)
        {
            int q = extMotionDim  + (j*motionDim);
            Fnew.block<6, 6>(q-motionDim, q).setIdentity();
            j++;
        }

        state.supportFrameData.erase(idx);
    }
    else
    {
        // remove landmark
        int ldmkIdx = idx - landmarkIDsOffset;
        if (state.landmarkData.find(ldmkIdx) == state.landmarkData.end())
        {
            std::cerr << "[DILIGENT::removeContactOrLandmark] landmark does not exist" << std::endl;
            return false;
        }

        int j = 0;
        for (auto& iter : state.supportFrameData)
        {
            int q = extMotionDim  + (j*motionDim);
            Fnew.block<6, 6>(q, q).setIdentity();
            j++;
        }

        for (auto& iter : state.landmarkData)
        {
            int q = extMotionDim  + (j*motionDim);
            if (iter.first < ldmkIdx)
            {
                Fnew.block<6, 6>(q, q).setIdentity();
            }
            else  if (iter.first > ldmkIdx)
            {
                Fnew.block<6, 6>(q-motionDim, q).setIdentity();
            }
            j++;
        }

        state.landmarkData.erase(ldmkIdx);
    }

    if (estimateBias)
    {
        Fnew.bottomRightCorner<6, 6>().setIdentity();
    }

    // update covariance
    P = Fnew * P * Fnew.transpose();
    return true;
}
