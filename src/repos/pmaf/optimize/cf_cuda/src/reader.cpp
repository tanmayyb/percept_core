// basic
#include  <iostream>
#include  <vector>
#include  <string>
#include  <algorithm>

// file reading
#define RYML_SINGLE_HDR_DEFINE_NOW
#include  <ryml_all.hpp>
#include  <fstream>
#include  <sstream>

// cuda-mode obstacles class
#include "obstacles.hpp" 


std::string filePath = "../assets/dual_arms_static3.yaml";


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


int loadObstacles(std::vector<Obstacle>* obstacles){
  // read YAML and load obstacles into vector

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

