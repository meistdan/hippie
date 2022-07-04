
# Try to find HIP driver API library and include path.
# Once done this will define
#
# HIP_DRIVER_API_FOUND
# HIP_DRIVER_API_LIBRARY
# HIP_INCLUDE_DIR
#

# Add HIP directory to prefix path.
if(DEFINED ENV{HIP_DIR})
  	set(HIP_DIR $ENV{HIP_DIR})
endif()
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${HIP_DIR})

if (WIN32)
	find_path(HIP_INCLUDE_DIR hip/hip_runtime_api.h
		${HIP_DIR}/include
		DOC "The directory where hip/hip_runtime_api.h resides"
	)
  	set(HIP_LIB_PATH ${HIP_DIR}/lib)
  	find_library(HIP_DRIVER_API_LIBRARY
		NAMES amdhip64
		PATHS
		${HIP_LIB_PATH}
		DOC "The HIP driver API library"
	)
else (WIN32)
	find_path(HIP_INCLUDE_DIR hip/hip_runtime_api.h
		/usr/include
		/usr/local/include
		/sw/include
		/opt/local/include
		DOC "The directory where hip/hip_runtime_api.h resides")
	find_library(HIP_DRIVER_API_LIBRARY
		NAMES amdhip64
		PATHS
		/usr/lib
		/usr/lib64
		/usr/local/lib
		/usr/local/lib64
		/sw/lib
		/opt/local/lib
		${HIP_DIR}/lib
		${HIP_DIR}/lib64
		DOC "The HIP driver API library")
endif (WIN32)

if (HIP_DRIVER_API_LIBRARY)
	set(HIP_DRIVER_API_FOUND 1 CACHE STRING "Set to 1 if HIP driver API lib. is found, 0 otherwise")
else (HIP_DRIVER_API_LIBRARY)
	set(HIP_DRIVER_API_FOUND 0 CACHE STRING "Set to 1 if HIP driver API lib. is found, 0 otherwise")
endif (HIP_DRIVER_API_LIBRARY)

mark_as_advanced(HIP_DRIVER_API_FOUND)
