#include <iostream>
#include <string>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>

#include <ament_index_cpp/get_package_share_directory.hpp>


int main() {
    // 1. Path to URDF
		std::string pkg_share_dir = ament_index_cpp::get_package_share_directory("percept");
		const std::string urdf_filename = "/assets/panda/panda.urdf";

    // 2. Load Model and Create Data Cache
    pinocchio::Model model;
    pinocchio::urdf::buildModel(pkg_share_dir+urdf_filename, model);
    pinocchio::Data data(model);

    // 3. Define Input States
    // Joint positions (q) for Panda (7 DOF)
    Eigen::VectorXd q = Eigen::VectorXd::Zero(model.nq);
    q << 0.1, 0.2, 0.0, -1.5, 0.0, 1.5, 0.7; 

    // Base pose in world frame (SE3: Rotation Matrix + Translation Vector)
    pinocchio::SE3 base_pose = pinocchio::SE3::Identity();
    base_pose.translation() << 0.5, 0.0, 0.0; // Offset base by 0.5m in X
    
    // Apply base pose to the model root (Joint 1 placement relative to universe)
    model.jointPlacements[1] = base_pose * model.jointPlacements[1];

    // 4. Compute Kinematics
    // Computes joint placements based on q
    pinocchio::forwardKinematics(model, data, q);

    // Update all frame placements (includes links and sensors)
    pinocchio::updateFramePlacements(model, data);

    // 5. Extract Results
    for (size_t frame_id = 0; frame_id < model.frames.size(); ++frame_id) {
        const auto& frame = model.frames[frame_id];
        const pinocchio::SE3& pose = data.oMf[frame_id];

        std::cout << "Frame: " << frame.name 
                  << " | Translation: " << pose.translation().transpose() 
                  << std::endl;
    }

    return 0;
}