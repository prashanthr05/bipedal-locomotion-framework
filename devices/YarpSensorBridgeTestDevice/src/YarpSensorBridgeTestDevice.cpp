/**
 * @file YarpSensorBridgeTestDevice.cpp
 * @authors Prashanth Ramadoss
 * @copyright 2020 Istituto Italiano di Tecnologia (IIT). This software may be modified and
 * distributed under the terms of the GNU Lesser General Public License v2.1 or any later version.
 */

#include <BipedalLocomotion/YarpSensorBridgeTestDevice.h>
#include <BipedalLocomotion/YarpUtilities/Helper.h>
#include <yarp/os/LogStream.h>

using namespace BipedalLocomotion::YarpUtilities;
using namespace BipedalLocomotion::ParametersHandler;
using namespace BipedalLocomotion::RobotInterface;
using namespace BipedalLocomotion;

YarpSensorBridgeTestDevice::YarpSensorBridgeTestDevice(double period,
                                                       yarp::os::ShouldUseSystemClock useSystemClock)
: yarp::os::PeriodicThread(period, useSystemClock)
{
}


YarpSensorBridgeTestDevice::YarpSensorBridgeTestDevice()
        : yarp::os::PeriodicThread(0.01, yarp::os::ShouldUseSystemClock::No)
{
}

YarpSensorBridgeTestDevice::~YarpSensorBridgeTestDevice()
{
}


bool YarpSensorBridgeTestDevice::open(yarp::os::Searchable& config)
{
    YarpUtilities::getElementFromSearchable(config, "robot", m_robot);
    YarpUtilities::getElementFromSearchable(config, "port_prefix", m_portPrefix);

    double devicePeriod{0.01};

    if (YarpUtilities::getElementFromSearchable(config, "sampling_period_in_s", devicePeriod))
    {
        setPeriod(devicePeriod);
    }

    if (!setupRobotSensorBridge(config))
    {
        return false;
    }

    return true;
}

bool YarpSensorBridgeTestDevice::setupRobotSensorBridge(yarp::os::Searchable& config)
{
    auto bridgeConfig = config.findGroup("RobotSensorBridge");
    if (bridgeConfig.isNull())
    {
        yError() << "[YarpSensorBridgeTestDevice][setupRobotSensorBridge] Missing required group \"RobotSensorBridge\"";
        return false;
    }

    std::shared_ptr<YarpImplementation> originalHandler = std::make_shared<YarpImplementation>();
    originalHandler->set(bridgeConfig);

    m_robotSensorBridge = std::make_unique<YarpSensorBridge>();
    if (!m_robotSensorBridge->initialize(originalHandler))
    {
        yError() << "[YarpSensorBridgeTestDevice][setupRobotSensorBridge] Could not configure RobotSensorBridge";
        return false;
    }

    return true;
}


bool YarpSensorBridgeTestDevice::attachAll(const yarp::dev::PolyDriverList & poly)
{
    if (!m_robotSensorBridge->setDriversList(poly))
    {
        yError() << "[YarpSensorBridgeTestDevice][attachAll] Could not attach drivers list to sensor bridge";
        return false;
    }

    start();
    return true;
}


void YarpSensorBridgeTestDevice::run()
{
    auto start = std::chrono::high_resolution_clock::now();
    m_robotSensorBridge->advance();
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count();
    yInfo() << "YarpSensorBridge advance() took " <<  duration << " us.";

    Eigen::Matrix<double, 12, 1> imuMeasurement;
    double time;
    if (!m_robotSensorBridge->getIMUMeasurement("root_link_imu_acc", imuMeasurement, &time))
    {
        yError() << "Couldn't get IMU measurement";
    }
    else
    {
        std::cout << "IMU measurement: " << imuMeasurement.transpose() << std::endl;
    }
}


bool YarpSensorBridgeTestDevice::detachAll()
{
    std::lock_guard<std::mutex> guard(m_deviceMutex);
    if (isRunning())
    {
        stop();
    }

    return true;
}


bool YarpSensorBridgeTestDevice::close()
{
    std::lock_guard<std::mutex> guard(m_deviceMutex);
    return true;
}

