/**
 * @file ContactModel.h
 * @authors Giulio Romualdi
 * @copyright 2020 Istituto Italiano di Tecnologia (IIT). This software may be modified and
 * distributed under the terms of the GNU Lesser General Public License v2.1 or any later version.
 */

#ifndef BIPEDAL_LOCOMOTION_CONTROLLERS_CONTACT_MODELS_CONTACT_MODEL_H
#define BIPEDAL_LOCOMOTION_CONTROLLERS_CONTACT_MODELS_CONTACT_MODEL_H

#include <iDynTree/Core/Wrench.h>

namespace BipedalLocomotionControllers
{
namespace ContactModels
{
/**
 * ContactModel is a generic implementation of a contact model. It computes the contact wrench
 * between the robot and the environments
 */
class ContactModel
{
protected:
    iDynTree::Wrench m_contactWrench; /**< Contact wrench between the robot and the environment
                                         expressed in mixed representation */
    bool m_isContactWrenchComputed; /**< If true the contact wrench has been already computed */

    /**
     * Evaluate the contact wrench given a specific contact model
     */
    virtual void computeContactWrench() = 0;

public:
    /**
     * Get and compute (only if it is necessary) the contact wrench
     * @return the contact wrench expressed in mixed representation
     */
    const iDynTree::Wrench& getContactWrench();
};
} // namespace ContactModels
} // namespace BipedalLocomotionControllers

#endif // BIPEDAL_LOCOMOTION_CONTROLLERS_CONTACT_MODELS_CONTACT_MODEL_H