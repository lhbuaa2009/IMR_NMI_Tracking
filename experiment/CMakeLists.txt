#add_executable( helloslam helloslam.cpp)

#add_library(testPixelSelector testPixelSelector.cpp)
#add_executable(test test.cpp)

#target_link_libraries(testPixelSelector cameraintrinsic ${OpenCV_LIBS} ${Sophus_LIBRARIES})
#target_link_libraries(test testPixelSelector)

add_executable(test_framereader test_FrameReader.cpp)
target_link_libraries(test_framereader framereader ${OpenCV_LIBS})

add_executable(test_initializer test_initializer.cpp)
target_link_libraries(test_initializer initializer framereader ${thirdparty_libs})

add_executable(test_Tracker test_Tracker.cpp)
target_link_libraries(test_Tracker tracker framereader ${thirdparty_libs})

#add_executable(test_Pcl test_pcl.cpp)
#target_link_libraries(test_Pcl ${PCL_LIBRARIES})

add_executable(test_Resdistribution test_Resdistribution.cpp)
target_link_libraries(test_Resdistribution frame pixelselector framereader ${OpenCV_LIBS})

add_executable(test_GrayCentroid test_GrayCentroid.cpp)
target_link_libraries(test_GrayCentroid frame pixelselector framereader ${OpenCV_LIBS})

add_executable(test_TrackingTrajectory test_TrackingTrajectory.cpp)
target_link_libraries(test_TrackingTrajectory tracker framereader ${thirdparty_libs})

add_executable(test_Tracker2 test_Tracker2.cpp)
target_link_libraries(test_Tracker2 tracker framereader ${thirdparty_libs})

add_executable(test_Optimizer test_Optimizer.cpp)
target_link_libraries(test_Optimizer tracker framereader optimizer ${thirdparty_libs})
