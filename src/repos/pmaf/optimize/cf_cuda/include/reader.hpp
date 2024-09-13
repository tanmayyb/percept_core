#pragma once
#include  <ryml_all.hpp>
#include  <vector>
#include "obstacles.hpp" 


int loadObstacles(std::vector<Obstacle>* obstacles);

Obstacle create_obstacle_object(ryml::NodeRef obstacle);