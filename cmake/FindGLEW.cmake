
# Try to find GLEW library and include path.
# Once done this will define
#
# GLEW_FOUND
# GLEW_INCLUDE_DIR
# GLEW_LIBRARIES
# 

if (EXISTS $ENV{GLEW})	
	set(GLEW_ROOT_PATH $ENV{GLEW} CACHE PATH "Path to glew")
else (EXISTS $ENV{GLEW})
	message("WARNING: GLEW enviroment variable is not set")
	set(GLEW_ROOT_PATH "" CACHE PATH "Path to glew")
endif (EXISTS $ENV{GLEW})

if (WIN32)
	find_path(GLEW_INCLUDE_DIR GL/glew.h
		${GLEW_ROOT_PATH}/include
		C:/glew_64bit/include
		C:/glew_32bit/include
		lib/glew_64bit/include
		DOC "The directory where GL/glew.h resides"
	)
	set(GLEW_LIB_PATHS
		${GLEW_ROOT_PATH}/lib
		C:/glew_64bit/lib
		lib/glew_64bit/lib
	)
	find_library(GLEW_LIBRARY
		NAMES glew glew32 GLEW
		PATHS
		${GLEW_LIB_PATHS}
		DOC "The GLEW library"
	)
	set(GLEW_LIBRARIES ${GLEW_LIBRARY})
else (WIN32)
	find_path(GLEW_INCLUDE_DIR GL/glew.h
		/usr/include
		/usr/local/include
		/sw/include
		/opt/local/include
		DOC "The directory where GL/glew.h resides"
	)
	find_library(GLEW_LIBRARY
		NAMES GLEW glew
		PATHS
		/usr/lib
		/usr/lib64
		/usr/local/lib
		/usr/local/lib64
		/sw/lib
		/opt/local/lib
		DOC "The GLEW library"
	)
	set(GLEW_LIBRARIES ${GLEW_LIBRARY})
endif (WIN32)

if (GLEW_INCLUDE_DIR)
	set( GLEW_FOUND 1 CACHE STRING "Set to 1 if GLEW is found, 0 otherwise")
else (GLEW_INCLUDE_DIR)
	set( GLEW_FOUND 0 CACHE STRING "Set to 1 if GLEW is found, 0 otherwise")
endif (GLEW_INCLUDE_DIR)

mark_as_advanced( GLEW_FOUND )
