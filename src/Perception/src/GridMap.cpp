/**
 * @file GridMap.cpp
 * @authors Prashanth Ramadoss
 * @copyright 2020 Istituto Italiano di Tecnologia (IIT). This software may be modified and
 * distributed under the terms of the GNU Lesser General Public License v2.1 or any later version.
 */

#include <BipedalLocomotion/Perception/Features/GridMap.h>
#include <pcl/common/common.h>
#include <cmath>

using namespace BipedalLocomotion::Perception::Features;

struct GridMap2D::Impl 
{
    void prepareMap();
    Eigen::MatrixXi occupancyMap;
    double resolution{0.01};
    Eigen::Vector2d origin;
    Eigen::Vector2d dimensions;
    
    std::pair<int, int> getMapIndexFromPosition(Eigen::Ref<const Eigen::Vector2d> xy);
    bool checkIfPositionWithinMap(Eigen::Ref<const Eigen::Vector2d> xy);
};

GridMap2D::GridMap2D() : m_pimpl(std::make_unique<Impl>())
{
    m_pimpl->origin.setZero();
    m_pimpl->dimensions.setZero();
}

bool GridMap2D::Impl::checkIfPositionWithinMap(Eigen::Ref<const Eigen::Vector2d> xy)
{
    if ( (xy(0) > - dimensions(0)/2) &&  (xy(0) < dimensions(0)/2) &&
         (xy(1) > - dimensions(1)/2) &&  (xy(1) < dimensions(1)/2) )
    {
        return true;
    }
    
    return false;
}

bool GridMap2D::setGeometry(Eigen::Ref<const Eigen::Vector2d> originCoordinates,
                            Eigen::Ref<const Eigen::Vector2d> dimensionsInMeters,        
                            const double& resolutionInMeters)
{
    if (resolutionInMeters < 1e-4)
    {
        std::cout << "[GridMap2D::populateMapFromPointCloud] Resolution must be greater than 1e-4." << std::endl;
        return false;
    }
    
    int rows = static_cast<int>(std::ceil(dimensionsInMeters(0))/resolutionInMeters) + 1;
    int cols = static_cast<int>(std::ceil(dimensionsInMeters(1))/resolutionInMeters) + 1;
    
    if (rows > std::numeric_limits<int>::max() ||
        cols > std::numeric_limits<int>::max())
    {     
        std::cout << "[GridMap2D::populateMapFromPointCloud] Specified dimensions might cause overflow." << std::endl;
        return false;
    }
    
    m_pimpl->origin = originCoordinates;
    m_pimpl->dimensions = dimensionsInMeters;
    m_pimpl->resolution = resolutionInMeters;
    
    // setting 
    m_pimpl->occupancyMap = -Eigen::MatrixXi::Ones(rows, cols);
    
    
    return true;
}



template<typename PointType> 
bool GridMap2D::populateMapFromPointCloud(const typename pcl::PointCloud<PointType>::Ptr inCloud, 
                                          const double& resolution)
{
    if (inCloud == nullptr)
    {
        std::cerr << "[GridMap2D::populateMapFromPointCloud] Invalid input point cloud" << std::endl;
        return false;
    }    
                
    pcl::PointXYZ minPt, maxPt;
    pcl::getMinMax3D(*inCloud, minPt, maxPt);
    
    // get index of point in the map and fill the map
    
    return true;
}


