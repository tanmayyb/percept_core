// the usual
#include  <iostream>
#include  <vector>
#include  <algorithm>

// file reading
#include  <ryml_all.hpp>
#include  <fstream>
#include  <sstream>
#include  <string>

// time keeping
#include  <chrono>
#include "Obstacles.hpp"


// special thingies
#include <Eigen/Dense>


int main(){
  Obstacle obstacle;

  std::cout<<"obstacle: pos_x_:\t"<<obstacle.pos_x_<<std::endl;
  std::cout<<"obstacle: pos_y_:\t"<<obstacle.pos_y_<<std::endl;
  std::cout<<"obstacle: pos_z_:\t"<<obstacle.pos_z_<<std::endl;

  Eigen::Vector3d v{69.0f, 69.0f, 69.0f};
  obstacle.setPosition(v);

  std::cout<<std::endl;
  std::cout<<"obstacle: pos_x_:\t"<<obstacle.pos_x_<<std::endl;
  std::cout<<"obstacle: pos_y_:\t"<<obstacle.pos_y_<<std::endl;
  std::cout<<"obstacle: pos_z_:\t"<<obstacle.pos_z_<<std::endl;


  return 0;
}