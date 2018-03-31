
add_library(cameraintrinsic SHARED CameraIntrinsic.cpp)

add_library(frame Frame.cpp)
add_library(immaturepoint ImmaturePoint.cpp)
add_library(point Point.cpp)
add_library(residual Residual.cpp)

target_link_libraries(immaturepoint frame)
target_link_libraries(frame point immaturepoint cameraintrinsic)

target_link_libraries(point frame residual  immaturepoint)
target_link_libraries(residual frame point)

add_library(framereader FrameReader.cpp)
target_link_libraries(framereader frame)

add_library(pixelselector PixelSelector.cpp)
target_link_libraries(pixelselector frame)

add_library(initializer Intializer.cpp)
target_link_libraries(initializer pixelselector frame)

add_library(tracker Tracker.cpp)
target_link_libraries(tracker pixelselector frame residual)

add_library(optimizer Optimizer.cpp)
target_link_libraries(optimizer pixelselector frame ${g2o_libs})
