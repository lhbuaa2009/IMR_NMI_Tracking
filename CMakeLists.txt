cmake_minimum_required( VERSION 2.8 )
project( SLAM )

# 设置用debug还是release模式。debug允许断点，而release更快
set( CMAKE_BUILD_TYPE Release )
set( CMAKE_CXX_COMPILER "g++" )
 # 设置编译选项
# 允许c++11标准、O3优化、多线程。match选项可避免一些cpu上的问题
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -O3 -pthread" )

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# 常见依赖库：cv, eigen, pcl
find_package( OpenCV REQUIRED )
find_package( Eigen3 REQUIRED )
find_package( PCL 1.7 REQUIRED )
find_package( Sophus REQUIRED)
find_package( G2O REQUIRED )

# 头文件目录
include_directories(
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/include/test
${PROJECT_SOURCE_DIR}/experiment
${PROJECT_SOURCE_DIR}/Thirdparty/g2o_modified
${Sophus_INCLUDE_DIRS}
${PCL_INCLUDE_DIRS}
)

add_definitions(${PCL_DEFINITIONS})

# link目录
link_directories(
${PROJECT_SOURCE_DIR}/lib
${PROJECT_SOURCE_DIR}/Thirdparty/g2o_modified/lib
${PCL_LIBRARY_DIRS}
)


set( thirdparty_libs
     ${OpenCV_LIBS}
     ${PCL_LIBRARIES}
   )

set( g2o_libs
     g2o_core g2o_types_sba g2o_solver_csparse g2o_stuff g2o_csparse_extension
   )

# 二进制文件输出到bin 库输出到lib
set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib )


# 源文件目录
add_subdirectory( ${PROJECT_SOURCE_DIR}/src/ )
add_subdirectory( ${PROJECT_SOURCE_DIR}/experiment/ )
