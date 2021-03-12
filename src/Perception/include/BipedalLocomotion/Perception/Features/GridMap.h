/**
 * @file GridMap.h
 * @authors Prashanth Ramadoss
 * @copyright 2020 Istituto Italiano di Tecnologia (IIT). This software may be modified and
 * distributed under the terms of the GNU Lesser General Public License v2.1 or any later version.
 */

#ifndef BIPEDAL_LOCOMOTION_PERCEPTION_GRIDMAP_H
#define BIPEDAL_LOCOMOTION_PERCEPTION_GRIDMAP_H

#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace BipedalLocomotion {

namespace Perception {
    
namespace Features {

enum class Occupancy
{
    FREE,      // 0 
    OCCUPIED , // 1 
    UNKOWN,    // -1
    ROBOT      // 100
};

struct GridCell
{
    Occupancy occupancy{Occupancy::UNKOWN};
    double probability{1.0};  // currently unused
    double height{0.0};       // currently unused
    double heightStdDev{1.0}; // currently unused
};


class GridMap2D
{
    GridMap2D();
    ~GridMap2D();
    
    bool setGeometry(Eigen::Ref<const Eigen::Vector2d> origin,
                     Eigen::Ref<const Eigen::Vector2d> dimensions,        
                     const double& resolution);
    
    template <typename PointType>
    bool populateMapFromPointCloud(const typename pcl::PointCloud<PointType>::Ptr inCloud,
                                   const double& resolution);
    
    std::vector< std::vector<Eigen::Vector2d> > getObstacleVertices();
    Eigen::MatrixXd getMap();
    
    GridCell getCell(const int& x, const int& y);
    Occupancy getCellOccupancy(const int& x, const int& y);
    
private:
    struct Impl;
    std::unique_ptr<Impl> m_pimpl;    
};
    
} // namespace Features  
} // namespace Perception    
} // namespace BipedalLocomotion
  


#endif
