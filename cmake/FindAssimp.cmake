
# Try to find Assimp library and include path.
# Once done this will define
#
# ASSIMP_FOUND
# ASSIMP_INCLUDE_DIR
# ASSIMP_LIBRARIES
#

if (EXISTS $ENV{ASSIMP})	
	set (ASSIMP_ROOT_PATH $ENV{ASSIMP} CACHE PATH "Path to assimp")
else (EXISTS $ENV{ASSIMP})
	message("WARNING: ASSIMP enviroment variable is not set")
	set (ASSIMP_ROOT_PATH "" CACHE PATH "Path to assimp")
endif (EXISTS $ENV{ASSIMP}) 

if (WIN32)
	find_path(ASSIMP_INCLUDE_DIR assimp/scene.h
		${ASSIMP_ROOT_PATH}/include
		C:/assimp_64bit/include
		C:/assimp_32bit/include
		lib/assimp_64bit/include
		DOC "The directory where assimp/scene.h resides"
	)
	set(ASSIMP_LIB_PATHS
		${ASSIMP_ROOT_PATH}/lib
		C:/assimp_64bit/lib
		lib/assimp_64bit/lib
	)
	find_library(ASSIMP_LIBRARY
		NAMES assimp
		PATHS
		${ASSIMP_LIB_PATHS}
		DOC "The Assimp library"
	)
	set(ASSIMP_LIBRARIES ${ASSIMP_LIBRARY})
else (WIN32)
	find_path(ASSIMP_INCLUDE_DIR assimp/scene.h
		/usr/local/include/assimp/
		/usr/include/
	)
	set (ASSIMP_LIB_PATHS
		/usr/lib/
		/usr/lib32/
		/usr/lib64
		/usr/local/lib
		/usr/local/lib64
		/usr/lib/x86_64-linux-gnu
		/sw/lib
		/opt/local/lib
	)
  find_library(ASSIMP_LIBRARY
		NAMES assimp
		PATHS
		${ASSIMP_LIB_PATHS}
		DOC "The Assimp library"
	)
	set (ASSIMP_LIBRARIES
		${ASSIMP_LIBRARY}
	)
endif (WIN32)

if (ASSIMP_INCLUDE_DIR)
	set(ASSIMP_FOUND 1 CACHE STRING "Set to 1 if Assimp is found, 0 otherwise")
else (ASSIMP_INCLUDE_DIR)
	set(ASSIMP_FOUND 0 CACHE STRING "Set to 1 if Assimp is found, 0 otherwise")
endif (ASSIMP_INCLUDE_DIR)

mark_as_advanced(ASSIMP_FOUND)
