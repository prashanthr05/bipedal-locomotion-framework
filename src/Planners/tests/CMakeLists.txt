# Copyright (C) 2020 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

if (FRAMEWORK_HAS_Eigen3)
    add_bipedal_test(
        NAME ContactList
        SOURCES ContactListTest.cpp
        LINKS BipedalLocomotion::Contact Eigen3::Eigen)

endif()

add_bipedal_test(
    NAME ContactPhaseList
    SOURCES ContactPhaseListTest.cpp
    LINKS BipedalLocomotion::Contact)

add_bipedal_test(
    NAME ConvexHullHelper
    SOURCES ConvexHullHelperTest.cpp
    LINKS BipedalLocomotion::Planners)
