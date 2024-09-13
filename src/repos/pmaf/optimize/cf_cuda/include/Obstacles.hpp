#pragma once
#include  <Eigen/Dense>
#include <cuda_runtime.h>


class Obstacle {
  public:
    std::string name_;
    double rad_;
    double pos_x_,pos_y_,pos_z_;
    double vel_x_, vel_y_, vel_z_;

  __host__ __device__ Obstacle(): 
    name_{""}, pos_x_{0}, pos_y_{0}, pos_z_{0}, vel_x_{0}, vel_y_{0}, vel_z_{0}, rad_{0} {};

  __host__ __device__ Obstacle( const std::string name,  const Eigen::Vector3d pos,  const Eigen::Vector3d vel,  const double rad 
  ): name_{name}, pos_x_{pos.x()}, pos_y_{pos.y()}, pos_z_{pos.z()}, vel_x_{vel.x()}, vel_y_{vel.y()}, vel_z_{vel.z()}, rad_{rad} {};

  __host__ __device__ Obstacle(
    const Eigen::Vector3d pos, const double rad
  ): pos_x_{pos.x()}, pos_y_{pos.y()}, pos_z_{pos.z()}, rad_{rad}, vel_x_{0}, vel_y_{0}, vel_z_{0}, name_{""} {};
  
  __host__ __device__ Obstacle(
    const Eigen::Vector3d pos, const Eigen::Vector3d vel, const double rad
  ): pos_x_{pos.x()}, pos_y_{pos.y()}, pos_z_{pos.z()}, vel_x_{vel.x()}, vel_y_{vel.y()}, vel_z_{vel.z()}, rad_{rad}, name_{""} {};
  
  // getters    
  __host__ std::string getName() const { return name_; };
  __host__ Eigen::Vector3d getPosition() const { Eigen::Vector3d p{pos_x_, pos_y_, pos_z_}; return p; };
  __host__ Eigen::Vector3d getVelocity() const { Eigen::Vector3d v{vel_x_, vel_y_, vel_z_}; return v; };
  __host__ double getRadius() const { return rad_; };

  // setters
  __host__ void setPosition(Eigen::Vector3d pos) { pos_x_ = pos.x(); pos_y_ = pos.y(); pos_z_ = pos.z();}
  __host__ void setVelocity(Eigen::Vector3d vel) { vel_x_ = vel.x(); vel_y_ = vel.y(); vel_z_ = vel.z();}

};