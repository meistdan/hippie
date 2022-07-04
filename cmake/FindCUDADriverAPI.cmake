
# Try to find CUDA driver API library and include path.
# Once done this will define
#
# CUDA_DRIVER_API_FOUND
# CUDA_DRIVER_API_LIBRARY
# CUDA_RUNTIME_API_LIBRARY
# CUDA_NVRTC_LIBRARY
# CUDA_INCLUDE_DIR
#

find_package(CUDA)

if (WIN32)
	set(CUDA_LIB_PATH ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
  	find_library(CUDA_DRIVER_API_LIBRARY
		NAMES cuda
		PATHS
		${CUDA_LIB_PATH}
		DOC "The CUDA driver API library"
	)
	find_library(CUDA_RUNTIME_API_LIBRARY
		NAMES cudart
		PATHS
		${CUDA_LIB_PATH}
		DOC "The CUDA NVRTC library"
	)
	find_library(CUDA_NVRTC_LIBRARY
		NAMES nvrtc
		PATHS
		${CUDA_LIB_PATH}
		DOC "The CUDA NVRTC library"
	)
else (WIN32)
	find_library(CUDA_DRIVER_API_LIBRARY
		NAMES cuda
		PATHS
		/usr/lib
		/usr/lib64
		/usr/local/lib
		/usr/local/lib64
		/sw/lib
		/opt/local/lib
		${CUDA_TOOLKIT_ROOT_DIR}/lib
		${CUDA_TOOLKIT_ROOT_DIR}/lib64
		DOC "The CUDA driver API library"
	)
	find_library(CUDA_RUNTIME_API_LIBRARY
		NAMES cudart
		PATHS
		/usr/lib
		/usr/lib64
		/usr/local/lib
		/usr/local/lib64
		/sw/lib
		/opt/local/lib
		${CUDA_TOOLKIT_ROOT_DIR}/lib
		${CUDA_TOOLKIT_ROOT_DIR}/lib64
		DOC "The CUDA NVRTC library"
	)
	find_library(CUDA_NVRTC_LIBRARY
		NAMES nvrtc
		PATHS
		/usr/lib
		/usr/lib64
		/usr/local/lib
		/usr/local/lib64
		/sw/lib
		/opt/local/lib
		${CUDA_TOOLKIT_ROOT_DIR}/lib
		${CUDA_TOOLKIT_ROOT_DIR}/lib64
		DOC "The CUDA NVRTC library"
	)
endif (WIN32)

if (CUDA_DRIVER_API_LIBRARY)
	set(CUDA_DRIVER_API_FOUND 1 CACHE STRING "Set to 1 if CUDA driver API lib. is found, 0 otherwise")
else (CUDA_DRIVER_API_LIBRARY)
	set(CUDA_DRIVER_API_FOUND 0 CACHE STRING "Set to 1 if CUDA driver API lib. is found, 0 otherwise")
endif (CUDA_DRIVER_API_LIBRARY)

mark_as_advanced(CUDA_DRIVER_API_FOUND)
