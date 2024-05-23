cmake_minimum_required(VERSION 3.12.0)

project(eVision)

add_executable(${PROJECT_NAME} main.cpp)

add_subdirectory(ext/opencv)

target_include_directories(${PROJECT_NAME} 
    PUBLIC ext/opencv/include
    )

# target_link_libraries(${PROJECT_NAME}
#     PRIVATE ext/opencv/src
# )
