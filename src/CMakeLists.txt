cmake_minimum_required(VERSION 3.4)

project(grasp_object_pybind)

# Find Python and Pybind11
find_package(Python3 COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# Set the path to libfranka
set(franka_DIR "/usr/local/lib/cmake/franka")  # Adjust this path as needed

# Find Franka
find_package(Franka REQUIRED)


# Find Threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Add the Pybind11 module
pybind11_add_module(grasp_object_pybind MODULE grasp_object.cpp)

# Include directories for Pybind11, Franka, Eigen3, and Poco
target_include_directories(grasp_object_pybind PRIVATE 
    ${PYTHON_INCLUDE_DIRS} 
    ${pybind11_INCLUDE_DIRS} 
    ${franka_INCLUDE_DIRS} 
)

# Link libraries
target_link_libraries(grasp_object_pybind PRIVATE 
    Franka::Franka 
    Threads::Threads
)



