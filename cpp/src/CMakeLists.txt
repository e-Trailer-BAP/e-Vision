cmake_minimum_required(VERSION 3.10.0)

#Set the project name
project(eVision)

# Source files
set(SOURCE_FILES main.cpp)

#Specify the executable
add_executable(${PROJECT_NAME} ${SOURCE_FILES})



#External libraries
add_subdirectory(ext/gstreamer)
add_subdirectory(ext/opencv)
# add_subdirectory(ext/opencv_contrib)

include_directories(${PROJECT_NAME} 
    ext/gstreamer/include
    ext/opencv/include
    )

# target_link_directories(${PROJECT_NAME}
#     PRIVATE ext/opencv/lib
# )

target_link_libraries(${PROJECT_NAME}
    PRIVATE gstreamer-1.0
    PRIVATE opencv_core
    # PRIVATE opencv_highgui
    PRIVATE opencv_imgproc
    # PRIVATE opencv_imgcodecs
    # PRIVATE opencv_videoio
    # PRIVATE opencv_video
    PRIVATE opencv_calib3d
    # PRIVATE opencv_features2d
    # PRIVATE opencv_flann
    # PRIVATE opencv_ml
    # PRIVATE opencv_objdetect
    # PRIVATE opencv_photo
    # PRIVATE opencv_shape
    # PRIVATE opencv_stitching
    # PRIVATE opencv_superres
    # PRIVATE opencv_videostab
    # PRIVATE opencv_dnn
    # PRIVATE opencv_dnn_objdetect
    # PRIVATE opencv_dpm
    # PRIVATE opencv_face
    # PRIVATE opencv_text
    # PRIVATE opencv_tracking
    # PRIVATE opencv_xfeatures2d
    # PRIVATE opencv_ximgproc
    # PRIVATE opencv_xobjdetect
    # PRIVATE opencv_aruco
    # PRIVATE opencv_bgsegm
    # PRIVATE opencv_bioinspired
    # PRIVATE opencv_ccalib
    # PRIVATE opencv_datasets
    # PRIVATE opencv_dnn_superres
    # PRIVATE opencv_dnn_text_detection
    # PRIVATE opencv_dnn_text_recognition
    )