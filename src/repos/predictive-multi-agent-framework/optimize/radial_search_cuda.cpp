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

extern void launch_radial_search_kernel(
    float* coords,
    float* target,
    float obstacle_rad,
    float detect_shell_rad,
    int num_points,
    int* output,
    int* output_count
);

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

void flattenCoordinates(
    const std::vector<Obstacle>& objects, 
    float* coords
) {
    int index = 0;
    for (const auto& obj : objects) {
        Eigen::Vector3d pos = obj.getPosition();
        coords[index++] = pos.x();
        coords[index++] = pos.y();
        coords[index++] = pos.z();
    }
}


int main(){
    std::cout<<"Program Start"<<std::endl;

    std::vector<Obstacle> obstacles;
    loadObstacles(&obstacles);

    int num_points = obstacles.size();
    float* coords = new float[num_points * 3];
    float* target = new float[3];
    target[0] = 0.0f;
    target[1] = 1.0f;
    target[2] = 0.5f;
    float obstacle_rad = 0.06;
    float detect_shell_rad = 0.3;
    int* output = new int[num_points * 3];
    int* output_count;



    // convert vector of Obstacles to Serializable datatype
    flattenCoordinates(obstacles, coords);

    // pass serialized data into kernel and get indices
    launch_radial_search_kernel(
        coords,
        target,
        obstacle_rad,
        detect_shell_rad,
        num_points,
        output,
        output_count
    );

    // create new vector of objects from old vector + indices
    // std::vector<Obstacle> obstacles_ = obstacles[indices]; // <<< these obstacles from radial search
    

    std::cout<<"Program Done"<<std::endl;

    return 0;
}