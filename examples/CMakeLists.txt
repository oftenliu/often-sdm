project(examples)
cmake_minimum_required(VERSION 2.8.7)

# The examples need a few additional dependencies (e.g. boost filesystem and OpenCV highgui):
find_package(OpenCV  REQUIRED core imgproc highgui objdetect)
message(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV lib dir found at ${OpenCV_LIB_DIR}")

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
set(BOOST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/boost_1.54")
set(BOOST_INCLUDE_DIRS "${BOOST_ROOT}/include")
set(BOOST_LIBRARY_DIRS "${BOOST_ROOT}/lib")

find_package(Boost 1.54.0 COMPONENTS system filesystem program_options REQUIRED)
if(Boost_FOUND)
	message(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
else(Boost_FOUND)
	message(FATAL_ERROR "Boost not found")
endif()
include_directories(${Boost_INCLUDE_DIRS})
if(MSVC)
	# The standard find_package for boost on Win finds the dynamic libs, so for dynamic linking to boost we need to #define:
	add_definitions(-DBOOST_ALL_NO_LIB) # Don't use the automatic library linking by boost with VS (#pragma ...). Instead, we specify everything here in cmake.
	add_definitions(-DBOOST_ALL_DYN_LINK) # Link against the dynamic boost lib - needs to match with the version that find_package finds.
endif()

# Simple example:
add_executable(simple_function simple_function.cpp)
target_link_libraries(simple_function pthread ${OpenCV_LIBS})
set_target_properties(simple_function PROPERTIES FOLDER "examples")

# Landmark detection example:
set(SOURCE
	landmark_detection.cpp
)
set(HEADERS
	hog.h
	#hog.c # used as a header, we don't want CMake to see it as a file it has to compile
)	
add_executable(landmark_detection ${SOURCE} ${HEADERS})
target_link_libraries(landmark_detection pthread ${OpenCV_LIBS} ${Boost_LIBRARIES})
set_target_properties(landmark_detection PROPERTIES FOLDER "examples")

# Pose estimation example:
add_executable(pose_estimation pose_estimation.cpp)
target_link_libraries(pose_estimation pthread ${OpenCV_LIBS} ${Boost_LIBRARIES})
set_target_properties(pose_estimation PROPERTIES FOLDER "examples")


add_executable(test_model test.cpp)
target_link_libraries(test_model pthread ${OpenCV_LIBS} ${Boost_LIBRARIES})
set_target_properties(test_model PROPERTIES FOLDER "examples")



add_executable(gen_box gen_box_data.cpp)
target_link_libraries(gen_box pthread ${OpenCV_LIBS} ${Boost_LIBRARIES})
set_target_properties(gen_box PROPERTIES FOLDER "examples")


# install target:
install(TARGETS simple_function DESTINATION bin)

install(TARGETS landmark_detection DESTINATION bin)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/examples/data/ DESTINATION bin/data/examples)

install(TARGETS pose_estimation DESTINATION bin)
