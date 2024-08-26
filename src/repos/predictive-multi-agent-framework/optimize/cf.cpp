#include  <iostream>
#include  <vector>
#include  <algorithm>
#include  <Eigen/Dense>

#define RYML_SINGLE_HDR_DEFINE_NOW
#include  <ryml_all.hpp>
#include  <fstream>
#include  <sstream>
#include  <string>
#include  <chrono>



/*


 /$$$$$$$                                       /$$$$$$  /$$                             
| $$__  $$                                     /$$__  $$| $$                             
| $$  \ $$  /$$$$$$   /$$$$$$$  /$$$$$$       | $$  \__/| $$  /$$$$$$   /$$$$$$$ /$$$$$$$
| $$$$$$$  |____  $$ /$$_____/ /$$__  $$      | $$      | $$ |____  $$ /$$_____//$$_____/
| $$__  $$  /$$$$$$$|  $$$$$$ | $$$$$$$$      | $$      | $$  /$$$$$$$|  $$$$$$|  $$$$$$ 
| $$  \ $$ /$$__  $$ \____  $$| $$_____/      | $$    $$| $$ /$$__  $$ \____  $$\____  $$
| $$$$$$$/|  $$$$$$$ /$$$$$$$/|  $$$$$$$      |  $$$$$$/| $$|  $$$$$$$ /$$$$$$$//$$$$$$$/
|_______/  \_______/|_______/  \_______/       \______/ |__/ \_______/|_______/|_______/ 
                                                                                         
                                                                                         
                                                                                         
*/

Eigen::Vector3d getLatestPosition() {
  Eigen::Vector3d pos{1.0,1.0,1.0};
  return pos;
}


Eigen::Vector3d getGoalPosition() {
  Eigen::Vector3d pos{0.5, 0.0, 0.7};
  return pos;
}

Eigen::Vector3d getVelocity() {
  Eigen::Vector3d pos{-0.1, -0.2, -0.2};
  return pos;
}

class Obstacle {
   private:
    std::string name_;
    Eigen::Vector3d pos_;
    Eigen::Vector3d vel_;
    double rad_;

   public:
    Obstacle(
      const std::string name, 
      const Eigen::Vector3d pos, 
      const Eigen::Vector3d vel, 
      const double rad
    ): name_{name}, pos_{pos}, vel_{vel}, rad_{rad} {};
    
    Obstacle(
      const Eigen::Vector3d pos, 
      const double rad
    ): pos_{pos}, rad_{rad}, vel_{0, 0, 0}, name_{""} {};
    
    Obstacle(
      const Eigen::Vector3d pos, 
      const Eigen::Vector3d vel, 
      const double rad
    ): pos_{pos}, rad_{rad}, vel_{vel}, name_{""} {};
    
    Obstacle(): pos_{0, 0, 0}, rad_{0}, vel_{0, 0, 0}, name_{""} {};

    // getters    
    std::string getName() const { return name_; };
    Eigen::Vector3d getPosition() const { return pos_; };
    Eigen::Vector3d getVelocity() const { return vel_; };
    double getRadius() const { return rad_; };

    // setters
    void setPosition(Eigen::Vector3d pos) { pos_ = pos; }
    void setVelocity(Eigen::Vector3d vel) { vel_ = vel; }

};


/*


 /$$                                 /$$        /$$$$$$  /$$                   /$$                         /$$                    
| $$                                | $$       /$$__  $$| $$                  | $$                        | $$                    
| $$        /$$$$$$   /$$$$$$   /$$$$$$$      | $$  \ $$| $$$$$$$   /$$$$$$$ /$$$$$$    /$$$$$$   /$$$$$$$| $$  /$$$$$$   /$$$$$$$
| $$       /$$__  $$ |____  $$ /$$__  $$      | $$  | $$| $$__  $$ /$$_____/|_  $$_/   |____  $$ /$$_____/| $$ /$$__  $$ /$$_____/
| $$      | $$  \ $$  /$$$$$$$| $$  | $$      | $$  | $$| $$  \ $$|  $$$$$$   | $$      /$$$$$$$| $$      | $$| $$$$$$$$|  $$$$$$ 
| $$      | $$  | $$ /$$__  $$| $$  | $$      | $$  | $$| $$  | $$ \____  $$  | $$ /$$ /$$__  $$| $$      | $$| $$_____/ \____  $$
| $$$$$$$$|  $$$$$$/|  $$$$$$$|  $$$$$$$      |  $$$$$$/| $$$$$$$/ /$$$$$$$/  |  $$$$/|  $$$$$$$|  $$$$$$$| $$|  $$$$$$$ /$$$$$$$/
|________/ \______/  \_______/ \_______/       \______/ |_______/ |_______/    \___/   \_______/ \_______/|__/ \_______/|_______/ 
                                                                                                                                  
                                                                                                                                  
                                                                                                                                  
*/

Obstacle create_obstacle_object(
  ryml::NodeRef obstacle
){
  Eigen::Vector3d pos;
  Eigen::Vector3d vel;
  double rad;
  int i=0;
  // read position
  for (auto item : obstacle["pos"].children()){
    std::string str_value(item.val().begin(), item.val().end());
    pos(i) = static_cast<double>(std::stod(str_value));
    i++;    
  }
  i=0;
  // read velocity
  for (auto item : obstacle["vel"].children()){
    std::string str_value(item.val().begin(), item.val().end());
    vel(i) = static_cast<double>(std::stod(str_value));
    i++;
  }
  // read radius
  auto item = obstacle["radius"];
  std::string str_value(item.val().begin(), item.val().end());
  rad = std::stod(str_value);

  Obstacle obj(pos, vel, rad);
  return obj;
}


// read YAML and load obstacles into vector
int loadObstacles(std::vector<Obstacle>* obstacles){
  std::string filePath = "dual_arms_static3.yaml";

  // open and read into buffer
  std::ifstream fileStream(filePath);
  if (!fileStream.is_open()) {
    std::cerr << "Failed to open the file: " << filePath << std::endl;
    return 1;
  }
  std::stringstream buffer;
  buffer << fileStream.rdbuf();
  std::string contents = buffer.str();
  fileStream.close();

  // construct yaml tree
  ryml::Tree tree = ryml::parse_in_arena(ryml::to_csubstr(contents));
  ryml::NodeRef obstacles_ = tree["bimanual_planning"]["obstacles"];

  // read and create obstacles
  for (auto obstacle:obstacles_)
    (*obstacles).push_back(create_obstacle_object(obstacle));

  return 0;
}



/*


  /$$$$$$  /$$                           /$$$$$$$$                                     
 /$$__  $$|__/                          | $$_____/                                     
| $$  \__/ /$$  /$$$$$$   /$$$$$$$      | $$     /$$$$$$   /$$$$$$   /$$$$$$$  /$$$$$$ 
| $$      | $$ /$$__  $$ /$$_____/      | $$$$$ /$$__  $$ /$$__  $$ /$$_____/ /$$__  $$
| $$      | $$| $$  \__/| $$            | $$__/| $$  \ $$| $$  \__/| $$      | $$$$$$$$
| $$    $$| $$| $$      | $$            | $$   | $$  | $$| $$      | $$      | $$_____/
|  $$$$$$/| $$| $$      |  $$$$$$$      | $$   |  $$$$$$/| $$      |  $$$$$$$|  $$$$$$$
 \______/ |__/|__/       \_______/      |__/    \______/ |__/       \_______/ \_______/
                                                                                       
                                                                                       
                                                                                       
*/



// // GoalObstacleHeuristicCfAgent
Eigen::Vector3d currentVector(
    const Eigen::Vector3d agent_pos, 
    const Eigen::Vector3d agent_vel,
    const Eigen::Vector3d goal_pos, 
    const std::vector<Obstacle> &obstacles,
    const int obstacle_id,
    const std::vector<Eigen::Vector3d> field_rotation_vecs) {
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
) {
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
  Eigen::Vector3d g_pos_ = getGoalPosition(); // goal_pos
  Eigen::Vector3d vel_ = getVelocity();
  const double detect_shell_rad_ = 0.1;
  const double rad_ = 0.5;
  double min_obs_dist_ = detect_shell_rad_; 
  std::vector<bool> known_obstacles_(obstacles.size(), false);
  std::vector<Eigen::Vector3d> field_rotation_vecs_(obstacles.size());


  auto start = std::chrono::high_resolution_clock::now();
  // optimize below this
  Eigen::Vector3d goal_vec{g_pos_ - getLatestPosition()};
  for (int i = 0; i < obstacles.size() - 1; i++) {

    Eigen::Vector3d robot_obstacle_vec{obstacles.at(i).getPosition() -
      getLatestPosition()};
    Eigen::Vector3d rel_vel{vel_ - obstacles.at(i).getVelocity()};

    if (robot_obstacle_vec.normalized().dot(goal_vec.normalized()) < -0.01 &&
      robot_obstacle_vec.dot(rel_vel) < -0.01) {
      continue;
    }
    double dist_obs{robot_obstacle_vec.norm() -
                    (rad_ + obstacles.at(i).getRadius())};

    dist_obs = std::max(dist_obs, 1e-5);
    if (dist_obs < min_obs_dist_) {
      min_obs_dist_ = dist_obs;
    }

    Eigen::Vector3d curr_force{0.0, 0.0, 0.0};
    Eigen::Vector3d current;


    if (dist_obs < detect_shell_rad_) {
      if (!known_obstacles_.at(i)) {
        field_rotation_vecs_.at(i) = calculateRotationVector(
            getLatestPosition(), 
            g_pos_, 
            obstacles, i
          );
        known_obstacles_.at(i) = true;
      }

      double vel_norm = rel_vel.norm();
      if (vel_norm != 0) {
        Eigen::Vector3d normalized_vel = rel_vel / vel_norm;
        current = currentVector(
          getLatestPosition(), rel_vel, getGoalPosition(),
          obstacles, i, field_rotation_vecs_);
        curr_force = (k_circ / pow(dist_obs, 2)) *
          normalized_vel.cross(current.cross(normalized_vel));
      }
    }

    force_ += curr_force;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "Function execution time: " << duration.count() << " seconds" << std::endl;

  std::cout<<force_<<std::endl;
}


/*


 /$$      /$$           /$$                 /$$$$$$$                                                               
| $$$    /$$$          |__/                | $$__  $$                                                              
| $$$$  /$$$$  /$$$$$$  /$$ /$$$$$$$       | $$  \ $$ /$$$$$$   /$$$$$$   /$$$$$$   /$$$$$$  /$$$$$$  /$$$$$$/$$$$ 
| $$ $$/$$ $$ |____  $$| $$| $$__  $$      | $$$$$$$//$$__  $$ /$$__  $$ /$$__  $$ /$$__  $$|____  $$| $$_  $$_  $$
| $$  $$$| $$  /$$$$$$$| $$| $$  \ $$      | $$____/| $$  \__/| $$  \ $$| $$  \ $$| $$  \__/ /$$$$$$$| $$ \ $$ \ $$
| $$\  $ | $$ /$$__  $$| $$| $$  | $$      | $$     | $$      | $$  | $$| $$  | $$| $$      /$$__  $$| $$ | $$ | $$
| $$ \/  | $$|  $$$$$$$| $$| $$  | $$      | $$     | $$      |  $$$$$$/|  $$$$$$$| $$     |  $$$$$$$| $$ | $$ | $$
|__/     |__/ \_______/|__/|__/  |__/      |__/     |__/       \______/  \____  $$|__/      \_______/|__/ |__/ |__/
                                                                         /$$  \ $$                                 
                                                                        |  $$$$$$/                                 
                                                                         \______/                                  
*/

int main(){
  std::vector<Obstacle> obstacles;

  loadObstacles(&obstacles);
  circForce(obstacles, 0.025);


  std::cout<<"Program Done"<<std::endl;

  return 0;
}