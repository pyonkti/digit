#include <pybind11/pybind11.h>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include <franka/exception.h>
#include <franka/gripper.h>

int grasp_object(char* ip_address, int homing_flag, char* width) {
  try {
    franka::Gripper gripper(ip_address);
    double grasping_width = std::stod(width);

    std::string homing_flag_str = std::to_string(homing_flag);
    std::stringstream ss(homing_flag_str);
    bool homing;
    if (!(ss >> homing)) {
      std::cerr << "<homing> can be 0 or 1." << std::endl;
      return -1;
    }

    if (homing) {
      // Do a homing in order to estimate the maximum grasping width with the current fingers.
      gripper.homing();
    }

    // Check for the maximum grasping width.
    franka::GripperState gripper_state = gripper.readOnce();
    if (gripper_state.max_width < grasping_width) {
      std::cout << "Object is too large for the current fingers on the gripper." << std::endl;
      return -1;
    }

    // Grasp the object.
    if (!gripper.grasp(grasping_width, 1, 100000)) {
      std::cout << "Failed to grasp object." << std::endl;
      return -1;
    }

  } catch (franka::Exception const& e) {
    std::cout << e.what() << std::endl;
    return -1;
  }

  return 0;
}

int open_gripper(char* ip_address){
try { 
    franka::Gripper gripper(ip_address);
    gripper.homing();

} catch (franka::Exception const& e) {
    std::cout << e.what() << std::endl;
    return -1;
  }

  return 0;
}
PYBIND11_MODULE(grasp_object_pybind, m) {
    m.def("grasp_object", &grasp_object, "Grasp an object using the Franka gripper");
    m.def("open_gripper", &open_gripper, "Open the Franka gripper");
}

