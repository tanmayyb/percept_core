#include<iostream>
#include<vector>
#include<algorithm>
#include <Eigen/Dense>

/*
getLatestPosition
calculateRotationVector
getGoalPosition
currentVector

normalized
dot
norm
pow
cross
*/



Eigen::Vector3d getLatestPosition() const {
  return pos_.back();
}


void CfAgent::setPosition(Eigen::Vector3d position) {
  pos_.clear();
  pos_.push_back(position);
}
void CfAgent::setVelocity(const Eigen::Vector3d &velocity) {
  double velocity_norm = velocity.norm();
  if (velocity_norm > vel_max_) {
    vel_ = (vel_max_ / velocity_norm) * velocity;
  } else {
    vel_ = velocity;
  }
}


class Obstacle {
   private:
    std::string name_;
    Eigen::Vector3d pos_;
    Eigen::Vector3d vel_;
    double rad_;

   public:
    Obstacle(const std::string name, const Eigen::Vector3d pos,
             const Eigen::Vector3d vel, const double rad)
        : name_{name}, pos_{pos}, vel_{vel}, rad_{rad} {};
    Obstacle(const Eigen::Vector3d pos, const double rad)
        : pos_{pos}, rad_{rad}, vel_{0, 0, 0}, name_{""} {};
    Obstacle(const Eigen::Vector3d pos, const Eigen::Vector3d vel,
             const double rad)
        : pos_{pos}, rad_{rad}, vel_{vel}, name_{""} {};
    Obstacle() : pos_{0, 0, 0}, rad_{0}, vel_{0, 0, 0}, name_{""} {};
    std::string getName() const { return name_; };
    double getRadius() const { return rad_; };
    void setPosition(Eigen::Vector3d pos) { pos_ = pos; }
    void setVelocity(Eigen::Vector3d vel) { vel_ = vel; }
    Eigen::Vector3d getPosition() const { return pos_; };
    Eigen::Vector3d getVelocity() const { return vel_; };
};



// // GoalObstacleHeuristicCfAgent
Eigen::Vector3d currentVector(
    const Eigen::Vector3d agent_pos, 
    const Eigen::Vector3d agent_vel,
    const Eigen::Vector3d goal_pos, 
    const std::vector<Obstacle> &obstacles,
    const int obstacle_id,
    const std::vector<Eigen::Vector3d> field_rotation_vecs) const {
  Eigen::Vector3d cfagent_to_obs{obstacles[obstacle_id].getPosition() -
    agent_pos};
  cfagent_to_obs.normalize();
  Eigen::Vector3d current{
      cfagent_to_obs.cross(field_rotation_vecs.at(obstacle_id))};
  current.normalize();
  return current;
}

// GoalObstacleHeuristicCfAgent
Eigen::Vector3d calculateRotationVector(
  const Eigen::Vector3d agent_pos, 
  const Eigen::Vector3d goal_pos,
  const std::vector<Obstacle> &obstacles, 
  const int obstacle_id
) const {
  double min_dist_obs = 100.0;
  int closest_obstacle_it = 0;
  for (int i = 0; i < obstacles.size() - 1; i++) {
    if (i != obstacle_id) {
      double dist_obs{
          (obstacles[obstacle_id].getPosition() - obstacles[i].getPosition())
              .norm()};
      if (min_dist_obs > dist_obs) {
        min_dist_obs = dist_obs;
        closest_obstacle_it = i;
      }
    }
  }

  // Vector from active obstacle to the obstacle which is closest to the
  // active obstacle
  Eigen::Vector3d obstacle_vec = obstacles[closest_obstacle_it].getPosition() -
                                 obstacles[obstacle_id].getPosition();
  Eigen::Vector3d cfagent_to_obs{obstacles[obstacle_id].getPosition() -
                                 agent_pos};
  cfagent_to_obs.normalize();
  // Current vector is perpendicular to obstacle surface normal and shows in
  // opposite direction of obstacle_vec
  Eigen::Vector3d obst_current{
      (cfagent_to_obs * obstacle_vec.dot(cfagent_to_obs)) - obstacle_vec};
  Eigen::Vector3d goal_vec{goal_pos - agent_pos};
  Eigen::Vector3d goal_current{goal_vec -
                               cfagent_to_obs * (cfagent_to_obs.dot(goal_vec))};
  Eigen::Vector3d current{goal_current.normalized() +
                          obst_current.normalized()};

  if (current.norm() < 1e-10) {
    current << 0.0, 0.0, 1.0;
    // current = makeRandomVector();
  }
  current.normalize();
  Eigen::Vector3d rot_vec{current.cross(cfagent_to_obs)};
  rot_vec.normalize();
  return rot_vec;
}


void circForce(
  const std::vector<Obstacle> &obstacles,
  const double k_circ
){
  Eigen::Vector3d force_{0.0,0.0,0.0};
  Eigen::Vector3d g_pos_{0.5, 0.0, 0.7}; // goal_pos
  Eigen::Vector3d goal_vec{g_pos_ - getLatestPosition()};
  
  // optimize this
  for (int i = 0; i < obstacles.size() - 1; i++) {

    Eigen::Vector3d robot_obstacle_vec{obstacles.at(i).getPosition() -
      getLatestPosition()};
    Eigen::Vector3d vel_{0.0,0.0,0.0};
    Eigen::Vector3d rel_vel{vel_ - obstacles.at(i).getVelocity()};

    // wtf is this 1?
    if (robot_obstacle_vec.normalized().dot(goal_vec.normalized()) < -0.01 &&
      robot_obstacle_vec.dot(rel_vel) < -0.01) {
      continue;
    }

    double dist_obs{robot_obstacle_vec.norm() -
                    (rad_ + obstacles.at(i).getRadius())};
    // wtf is this 2?
    // min_obs_dist_ = detect_shell_rad
    // dist_obs = std::max(dist_obs, 1e-5);
    // if (dist_obs < min_obs_dist_) {
    //   min_obs_dist_ = dist_obs;
    // }


    // std::vector<bool> known_obstacles_;
    Eigen::Vector3d curr_force{0.0, 0.0, 0.0};
    Eigen::Vector3d current;
    float detect_shell_rad = 0.10;


    if (dist_obs < detect_shell_rad_) {
      if (!known_obstacles_.at(i)) {
        field_rotation_vecs_.at(i) = calculateRotationVector(
            getLatestPosition(), 
            getGoalPosition(), 
            obstacles, i
          );
        known_obstacles_.at(i) = true;
      }

      double vel_norm = rel_vel.norm();
      if (vel_norm != 0) {
        Eigen::Vector3d normalized_vel = rel_vel / vel_norm;
        current = currentVector(
          getLatestPosition(), rel_vel, 
          getGoalPosition(),
          obstacles, i, field_rotation_vecs_);
        curr_force = (k_circ / pow(dist_obs, 2)) *
          normalized_vel.cross(current.cross(normalized_vel));
      }
    }

    force_ += curr_force;
  }
}



int main(){
    circForce();

    std::cout<<"WahGwan"<<std::endl;

    return 0;
}