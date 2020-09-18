/**
 * @file YarpSensorBridgeTestDevice.h
 * @authors Prashanth Ramadoss
 * @copyright 2020 Istituto Italiano di Tecnologia (IIT). This software may be modified and
 * distributed under the terms of the GNU Lesser General Public License v2.1 or any later version.
 */

#ifndef BIPEDAL_LOCOMOTION_FRAMEWORK_YARP_SENSOR_BRIDGE_TEST_DEVICE_H
#define BIPEDAL_LOCOMOTION_FRAMEWORK_YARP_SENSOR_BRIDGE_TEST_DEVICE_H

#include <BipedalLocomotion/ParametersHandler/YarpImplementation.h>
#include <BipedalLocomotion/RobotInterface/YarpSensorBridge.h>

#include <yarp/os/PeriodicThread.h>
#include <yarp/dev/DeviceDriver.h>
#include <yarp/dev/Wrapper.h>
#include <yarp/os/ResourceFinder.h>

#include <mutex>
#include <memory>


namespace BipedalLocomotion
{
    class YarpSensorBridgeTestDevice;
}

class BipedalLocomotion::YarpSensorBridgeTestDevice : public yarp::dev::DeviceDriver,
                                                      public yarp::dev::IMultipleWrapper,
                                                      public yarp::os::PeriodicThread
{
public:
    explicit YarpSensorBridgeTestDevice(double period, yarp::os::ShouldUseSystemClock useSystemClock = yarp::os::ShouldUseSystemClock::No);
    YarpSensorBridgeTestDevice();
    ~YarpSensorBridgeTestDevice();

    virtual bool open(yarp::os::Searchable& config) final;
    virtual bool close() final;
    virtual bool attachAll(const yarp::dev::PolyDriverList & poly) final;
    virtual bool detachAll() final;
    virtual void run() final;

private:
    bool setupRobotSensorBridge(yarp::os::Searchable& config);
    std::string m_robot{"test"};
    std::string m_portPrefix{"/testYarpSensorBridge"};
    std::unique_ptr<BipedalLocomotion::RobotInterface::YarpSensorBridge> m_robotSensorBridge;
    std::mutex m_deviceMutex;
};



#endif //BIPEDAL_LOCOMOTION_FRAMEWORK_YARP_SENSOR_BRIDGE_TEST_DEVICE_H
