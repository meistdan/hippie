
# Try to find DevIL library and include path.
# Once done this will define
#
# DEVIL_FOUND
# DEVIL_INCLUDE_DIR
# DEVIL_LIBRARIES
#

if (EXISTS $ENV{DEVIL})	
	set(DEVIL_ROOT_PATH $ENV{DEVIL} CACHE PATH "Path to DevIL")
else (EXISTS $ENV{DEVIL})  
	message("WARNING: DEVIL enviroment variable is not set")
	set(DEVIL_ROOT_PATH "" CACHE PATH "Path to DevIL")
endif (EXISTS $ENV{DEVIL})

if (WIN32)
	find_path(DEVIL_INCLUDE_DIR IL/il.h 
		${DEVIL_ROOT_PATH}/include
		C:/devil_64bit/include
		C:/devil_32bit/include
		lib/devil_64bit/include
		DOC "The directory where IL/il.h resides"
	)
	set(DEVIL_LIB_PATHS
		${DEVIL_ROOT_PATH}/lib
		C:/devil_64bit/lib
		lib/devil_64bit/lib
	)
	find_library(DEVIL_IL_LIBRARY
		NAMES DevIL
		PATHS
		${DEVIL_LIB_PATHS}
		DOC "The DevIL library"
	)
	find_library(DEVIL_ILU_LIBRARY
		NAMES ILU
		PATHS
		${DEVIL_LIB_PATHS}
		DOC "The ILU library"
	)
	set(DEVIL_LIBRARIES 
		${DEVIL_IL_LIBRARY}
		${DEVIL_ILU_LIBRARY}
	)
else (WIN32)
	find_path(DEVIL_INCLUDE_DIR IL/il.h
		/usr/local/include/DevIL/
		/usr/include/
	)
	set (DEVIL_LIB_PATHS
		/usr/lib/
		/usr/lib32/
		/usr/lib64
		/usr/local/lib
		/usr/local/lib64
		/usr/lib/x86_64-linux-gnu
		/sw/lib
		/opt/local/lib
	)
  find_library(DEVIL_IL_LIBRARY
		NAMES IL
		PATHS
		${DEVIL_LIB_PATHS}
		DOC "The IL library"
	)
  find_library(DEVIL_ILU_LIBRARY
		NAMES ILU
		PATHS
		${DEVIL_LIB_PATHS}
		DOC "The ILU library"
	)
	set (DEVIL_LIBRARIES
		${DEVIL_IL_LIBRARY} 
		${DEVIL_ILU_LIBRARY}
	)
endif (WIN32)

if (DEVIL_INCLUDE_DIR)
	set(DEVIL_FOUND 1 CACHE STRING "Set to 1 if DevIL is found, 0 otherwise")
else (DEVIL_INCLUDE_DIR)
	set(DEVIL_FOUND 0 CACHE STRING "Set to 1 if DevIL is found, 0 otherwise")
endif (DEVIL_INCLUDE_DIR)

mark_as_advanced(DEVIL_FOUND)
