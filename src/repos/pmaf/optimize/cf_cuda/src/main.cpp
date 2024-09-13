// the usual
#include  <iostream>
#include  <vector>
#include  <algorithm>

// necessary evils
#include <Eigen/Dense>
#include "obstacles.hpp"
#include "reader.hpp"

// time keeping
#include  <chrono>

// cf cuda-mode
#include <cf.h>


int main(){
  // Obstacle obstacle;

  std::vector<Obstacle> obstacles;
  loadObstacles(&obstacles);
  int n_obstacles = obstacles.size();

  double detect_shell_rad_;
  double* goalPosition = new double[3];
  double* agentPosition = new double[3];
  double* agentVelocity = new double[3];
  double* force = new double[3];

  goalPosition[0] = 0.5; goalPosition[1] = 0.0; goalPosition[2] = 0.7;
  agentPosition[0] = 0.0; agentPosition[1] = 0.0; agentPosition[2] = 0.5;
  agentVelocity[0] = -0.1; agentVelocity[1] = -0.1; agentVelocity[2] = -0.1;
  force[0] = 0.0; force[1] = 0.0; force[2] = 0.0;
  
  int N = 5;
  for(int i=0; i<N;i++){

    detect_shell_rad_ = double(i)*double(i)/200.0;
 
    // goalPosition[0] = 0.5; 
    // goalPosition[1] = 0.0; 
    // goalPosition[2] = 0.7;
    launch_circForce_kernel(
      &obstacles, 
      n_obstacles,
      0.025d, 
      detect_shell_rad_,
      goalPosition,
      agentPosition,
      agentVelocity,
      force
    );
  }

  hello_world();

  return 0;
}