# Min. CMake version.
cmake_minimum_required(VERSION 2.8.12)

# A name of the project.
project(hippie)

# CUDA flag.
option(CUDA_USE_CUDA "Use CUDA instead of HIP" ON)

# Add custom modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Find external libraries.
find_package(OpenGL)
find_package(GLEW)
find_package(Assimp)
find_package(DevIL)
find_package(HIPDriverAPI)

# Find CUDA driver API.
if(${CUDA_USE_CUDA})
  find_package(CUDADriverAPI)
endif(${CUDA_USE_CUDA})

# Add QT5 directory to prefix path.
if(DEFINED ENV{QT5_DIR})
set(QT5_DIR $ENV{QT5_DIR})
endif()
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${QT5_DIR})

# QT Core.
find_package(Qt5Core)
if(${Qt5Core_FOUND})
  message(STATUS "--> Qt5Core found.")
else(${Qt5Core_FOUND})
  message(FATAL_ERROR "Qt5Core not found. Please set QT5_DIR.")
endif(${Qt5Core_FOUND})

# QT Widgets.
find_package(Qt5Widgets)
if(${Qt5Widgets_FOUND})
  message(STATUS "--> Qt5Widgets found.")
else(${Qt5Widgets_FOUND})
  message(FATAL_ERROR "Qt5Widgets not found.")
endif(${Qt5Widgets_FOUND})
  
# QT OpenGL.  
find_package(Qt5OpenGL)
if(${Qt5OpenGL_FOUND})
  message(STATUS "--> Qt5OpenGL found.")
else(${Qt5OpenGL_FOUND})
  message(FATAL_ERROR "Qt5OpenGL not found.")
endif(${Qt5OpenGL_FOUND})

if(${CUDA_USE_CUDA})
  add_compile_definitions(__HIP_PLATFORM_NVCC__)
  add_compile_definitions(__HIP_PLATFORM_NVIDIA__)
else(${CUDA_USE_CUDA})  
  add_compile_definitions(__HIP_PLATFORM_AMD__)
endif(${CUDA_USE_CUDA})

# Folder hierarchy.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Add source files.
add_subdirectory(src)