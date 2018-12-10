cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(vlo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Debug")
set(DTARGET_ARCHITECTURE generic)
#set(CMAKE_CXX_FLAGS "-O")

# submodules
#add_subdirectory(ORB_SLAM2) #Need to do this before include_directories

# Dependencies
find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)
find_package(PCL REQUIRED)
find_package(Ceres REQUIRED)
find_package(CGAL REQUIRED)
set(CGAL_INCLUDE_DIRS /usr/local/include)
find_package(GMP REQUIRED)
find_package(GFLAGS REQUIRED)
find_package(Boost REQUIRED)

#file(GLOB orb_sources ${CMAKE_SOURCE_DIR}/ORB_SLAM2/src/*.c ${CMAKE_SOURCE_DIR}/ORB_SLAM2/src/*.cc ${CMAKE_SOURCE_DIR}/ORB_SLAM2/src/*.cpp)
#file(GLOB orb_headers ${CMAKE_SOURCE_DIR}/ORB_SLAM2/include/*.h ${CMAKE_SOURCE_DIR}/ORB_SLAM2/include/*.hpp)
#set(ORB_SLAM2_LIBRARY orbslam)
#add_library(${ORB_SLAM2_LIBRARY} SHARED ${orb_headers} ${orb_sources})

# Headers
include_directories(
        include
        ${OpenCV_INCLUDE_DIRS}
        ${Pangolin_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        ${CGAL_INCLUDE_DIRS}
        #ORB_SLAM2
        #ORB_SLAM2/src
        #ORB_SLAM2/include
        ${GMP_INCLUDE_DIR}
        ${BOOST_INCLUDE_DIR}
        usr/local/include
        /usr/local/include/opencv
        /usr/local/include/opencv2
)

#add_library(vlo_utils
#        src/ComputPoseFromInterPoint.cpp
#        src/Geo.cpp
#        src/IncrementalStatistics.cpp
#        src/IntersectionLine.cpp
#        src/IntersectionPoint.cpp
#        src/LiDARPoint.cpp
#        src/Line3D.cpp
#        src/LineFitting.cpp
#        src/LBDLineMatcher.cpp
#        src/MapLine.cpp
#        src/MapPlane.cpp
#        src/MapPoint.cpp
#        src/Opt.cpp
#        src/ORBextractor.cc
#        src/ORBmatcher.cc
#        src/PlaneFitting.cpp
#        src/PlaneLineDetector.cpp
#        src/Tracker.cpp
#        src/utils.cpp
#        src/VLFrame.cpp
#        src/VLMap.cpp
#        )

MESSAGE(STATUS "****************************************************")
MESSAGE(STATUS ${OpenCV_LIBS} )
MESSAGE(STATUS "****************************************************")



#add_executable(test_mono_kitti src/mono_kitti.cpp)
#target_link_libraries(test_mono_kitti
#						${OpenCV_LIBS}
#						${Pangolin_LIBRARIES}
#						${PCL_LIBRARIES}
#						${CERES_LIBRARIES}
#						${CGAL_LIBRARY}
#						${GMP_LIBRARIES}
#						vlo_utils
#						${GLOG_LIBRARIES}
#						${GFLAGS_LIBRARIES}
#						/usr/local/lib/libceres.so
#						/usr/local/lib/libceres.so.1
#						/usr/local/lib/libopencv_line_descriptor.so
#						/usr/local/lib/libCGAL.so
#						/usr/local/lib/libCGAL.so.13
#						)

add_executable(LPO main.cpp)
target_link_libraries(LPO
        ${OpenCV_LIBS}
        ${Pangolin_LIBRARIES}
        ${PCL_LIBRARIES}
        ${CERES_LIBRARIES}
        ${CGAL_LIBRARY}
        ${GMP_LIBRARIES}
#        vlo_utils
        ${GLOG_LIBRARIES}
        ${GFLAGS_LIBRARIES}
        /usr/local/lib/libceres.so
        /usr/local/lib/libceres.so.1
        /usr/local/lib/libopencv_line_descriptor.so
        /usr/local/lib/libCGAL.so
        /usr/local/lib/libCGAL.so.13
        #						${CMAKE_SOURCE_DIR}/ORB_SLAM2/lib/libORB_SLAM2.so
        )

#add_executable(test_PlaneICP src/test_PlaneICP.cpp)
#target_link_libraries(test_PlaneICP
#						${OpenCV_LIBS}
#						${Pangolin_LIBRARIES}
#						${PCL_LIBRARIES}
#						${CERES_LIBRARIES}
#						${CGAL_LIBRARY}
#						${GMP_LIBRARIES}
#						vlo_utils
#						${GLOG_LIBRARIES}
#						${GFLAGS_LIBRARIES}
#						/usr/local/lib/libceres.so
#						/usr/local/lib/libceres.so.1
#						/usr/local/lib/libopencv_line_descriptor.so
#						/usr/local/lib/libCGAL.so
#						/usr/local/lib/libCGAL.so.13
#						)


