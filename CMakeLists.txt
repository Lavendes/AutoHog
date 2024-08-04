cmake_minimum_required(VERSION 3.16)
project(HomGateProject)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set C++ compiler flags
set(CMAKE_CXX_FLAGS "-march=native -O3 -g -funroll-loops -Wall -Wextra -pedantic -Wno-sign-compare")

# Add TFHEpp library
add_subdirectory(TFHEpp)

# Include directories
include_directories(TFHEpp/src)

# Find and link OpenMP
find_package(OpenMP)
if(OpenMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Add the executables
add_executable(HomGateEval HomGate.cpp)
add_executable(CircuitEval Circuit_eval.cpp)

# Link the TFHEpp library to the executables
target_link_libraries(HomGateEval tfhe++)
target_link_libraries(CircuitEval tfhe++)

# Link OpenMP if found
if(OpenMP_FOUND)
    target_link_libraries(HomGateEval ${OpenMP_LIBRARIES})
    target_link_libraries(CircuitEval ${OpenMP_LIBRARIES})
endif()
