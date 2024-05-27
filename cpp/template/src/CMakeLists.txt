cmake_minimum_required(VERSION 3.10.0)

#Set the project name
project(template)

# Source files
set(SOURCE_FILES main.cpp)

#Specify the executable
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

find_package(PkgConfig REQUIRED)
pkg_check_modules(gtk3 REQUIRED IMPORTED_TARGET gtk+-3.0)
pkg_check_modules(opencv REQUIRED IMPORTED_TARGET opencv4)
pkg_search_module(gstreamer REQUIRED IMPORTED_TARGET gstreamer-1.0)
pkg_search_module(glib REQUIRED IMPORTED_TARGET glib-2.0)
pkg_search_module(gstreamer-sdp REQUIRED IMPORTED_TARGET gstreamer-sdp-1.0>=1.4)
pkg_search_module(gstreamer-app REQUIRED IMPORTED_TARGET gstreamer-app-1.0>=1.4)
pkg_search_module(gstreamer-video REQUIRED IMPORTED_TARGET gstreamer-video-1.0>=1.4)
pkg_search_module(yaml-cpp REQUIRED IMPORTED_TARGET yaml-cpp)
# pkg_search_module(<LIB NICKNAME> REQUIRED IMPORTED_TARGET <LIB NAME> [>= <VERSION> !!optional] )

target_link_libraries(${PROJECT_NAME}
    PkgConfig::gtk3
    PkgConfig::opencv
    PkgConfig::gstreamer
    PkgConfig::glib
    PkgConfig::gstreamer-sdp
    PkgConfig::gstreamer-app
    PkgConfig::gstreamer-video
    PkgConfig::yaml-cpp
    # PkgConfig::<LIB NICKNAME>
)