set(ROOT_HDR
  Globals.h
)

set(ROOT_SRC
  Globals.cpp
  main.cpp
)

set(BENCHMARK_HDR
  benchmark/Benchmark.h
)

set(BENCHMARK_SRC
  benchmark/Benchmark.cpp
)

set(ENVIRONMENT_HDR
  environment/AppEnvironment.h
  environment/Environment.h
)

set(ENVIRONMENT_SRC
  environment/AppEnvironment.cpp
  environment/Environment.cpp
)

set(GPU_HDR
  gpu/Buffer.h
  gpu/HipCompiler.h
  gpu/HipKernel.h
  gpu/HipModule.h
  gpu/Texture.h
)

set(GPU_SRC
  gpu/Buffer.cpp
  gpu/HipCompiler.cpp
  gpu/HipKernel.cpp
  gpu/HipModule.cpp
  gpu/Texture.cpp
)

set(GUI_HDR
  gui/AboutDialog.h
  gui/HelpDialog.h
  gui/LightWindow.h
  gui/MainWindow.h
  gui/OpenSceneDialog.h
  gui/RenderParamsWindow.h
  gui/RenderWidget.h
)

set(RADIX_SORT_HDR
  radix_sort/RadixSortUtil.h
  radix_sort/Reduce.h
  radix_sort/Scan.h
  radix_sort/RadixSort.h
  radix_sort/RadixSortKernels.h
  radix_sort/RadixSortPolicy.h
)

set(RADIX_SORT_SRC
  radix_sort/RadixSort.cpp
  radix_sort/RadixSortKernels.cu
)

set(GUI_SRC
  gui/AboutDialog.cpp
  gui/HelpDialog.cpp
  gui/LightWindow.cpp
  gui/MainWindow.cpp
  gui/OpenSceneDialog.cpp
  gui/RenderParamsWindow.cpp
  gui/RenderWidget.cpp
)

set(RT_HDR
  rt/HipUtil.h
  rt/TaskQueue.h
)

set(RT_BVH_HDR
  rt/bvh/ATRBuilder.h
  rt/bvh/ATRBuilderKernels.h
  rt/bvh/BVH.h	
  rt/bvh/BVHBuilder.h
  rt/bvh/BVHCollapser.h
  rt/bvh/BVHCollapserKernels.h
  rt/bvh/BVHConverter.h
  rt/bvh/BVHExporter.h
  rt/bvh/BVHImporter.h
  rt/bvh/HipBVH.h
  rt/bvh/HipBVHKernels.h
  rt/bvh/HipBVHNode.h
  rt/bvh/HipBVHUtil.h
  rt/bvh/HLBVHBin.h
  rt/bvh/HLBVHBuilder.h
  rt/bvh/HLBVHBuilderKernels.h
  rt/bvh/InsertionBuilder.h
  rt/bvh/InsertionBuilderKernels.h
  rt/bvh/LBVHBuilder.h
  rt/bvh/LBVHBuilderKernels.h
  rt/bvh/PLOCBuilder.h
  rt/bvh/PLOCBuilderKernels.h
  rt/bvh/Presplitter.h
  rt/bvh/PresplitterKernels.h
  rt/bvh/TRBuilder.h
  rt/bvh/TRBuilderKernels.h
  rt/bvh/TRBuilderUtil.h
  rt/bvh/SBVHBuilder.h
)

set(RT_BVH_SRC
  rt/bvh/ATRBuilder.cpp
  rt/bvh/ATRBuilderKernels.cu
  rt/bvh/BVHCollapser.cpp
  rt/bvh/BVHCollapserKernels.cu
  rt/bvh/BVHConverter.cpp
  rt/bvh/HipBVH.cpp
  rt/bvh/HipBVHKernels.cu
  rt/bvh/HLBVHBuilder.cpp
  rt/bvh/HLBVHBuilderKernels.cu
  rt/bvh/InsertionBuilder.cpp
  rt/bvh/InsertionBuilderKernels.cu
  rt/bvh/LBVHBuilder.cpp
  rt/bvh/LBVHBuilderKernels.cu
  rt/bvh/PLOCBuilder.cpp
  rt/bvh/PLOCBuilderKernels.cu
  rt/bvh/Presplitter.cpp
  rt/bvh/PresplitterKernels.cu
  rt/bvh/TRBuilder.cpp
  rt/bvh/TRBuilderKernels.cu
  rt/bvh/SBVHBuilder.cpp
)

set(RT_RAY_HDR
  rt/ray/PixelTable.h
  rt/ray/Ray.h
  rt/ray/RayBuffer.h
  rt/ray/RayBufferKernels.h
  rt/ray/RayGen.h
  rt/ray/RayGenKernels.h
)

set(RT_RAY_SRC
  rt/ray/PixelTable.cpp
  rt/ray/RayBuffer.cpp
  rt/ray/RayBufferKernels.cu
  rt/ray/RayGen.cpp
  rt/ray/RayGenKernels.cu
)

set(RT_RENDERER_HDR
  rt/renderer/Renderer.h
  rt/renderer/RendererKernels.h
)

set(RT_RENDERER_SRC
  rt/renderer/Renderer.cpp
  rt/renderer/RendererKernels.cu
)

set(RT_SCENE_HDR
  rt/scene/Camera.h
  rt/scene/DynamicScene.h
  rt/scene/EnvironmentMap.h
  rt/scene/Interpolator.h
  rt/scene/InterpolatorKernels.h
  rt/scene/Material.h
  rt/scene/SceneLoader.h
  rt/scene/Scene.h
  rt/scene/TextureAtlas.h
)

set(RT_SCENE_SRC
  rt/scene/Camera.cpp
  rt/scene/DynamicScene.cpp
  rt/scene/EnvironmentMap.cpp
  rt/scene/Interpolator.cpp
  rt/scene/InterpolatorKernels.cu
  rt/scene/SceneLoader.cpp
  rt/scene/Scene.cpp
  rt/scene/TextureAtlas.cpp
)

set(RT_TRACER_HDR
  rt/tracer/HipTracer.h
  rt/tracer/HipTracerKernels.h
)

set(RT_TRACER_SRC
  rt/tracer/HipTracer.cpp
  rt/tracer/HipTracerBinKernels.cu
  rt/tracer/HipTracerQuadKernels.cu
  rt/tracer/HipTracerOctKernels.cu
)

set(SHADER_SRC
  shader/fragment.fs
  shader/vertex.vs
)

set(UTIL_HDR
  util/AABB.h
  util/AssimpImporter.h
  util/BinExporter.h
  util/BinImporter.h
  util/Hash.h
  util/ImageExporter.h
  util/ImageImporter.h
  util/Logger.h
  util/Math.h
  util/Random.h
  util/SceneExporter.h
  util/SceneImporter.h
)

set(UTIL_SRC
  util/AssimpImporter.cpp
  util/BinExporter.cpp
  util/BinImporter.cpp
  util/Hash.cpp
  util/ImageExporter.cpp
  util/ImageImporter.cpp
  util/Logger.cpp
  util/Math.cpp
)

set(MOC_HDR
  gui/AboutDialog.h
  gui/HelpDialog.h
  gui/LightWindow.h
  gui/MainWindow.h
  gui/OpenSceneDialog.h
  gui/RenderParamsWindow.h
  gui/RenderWidget.h
)

set(UIS
  gui/AboutDialog.ui
  gui/HelpDialog.ui
  gui/LightWindow.ui
  gui/MainWindow.ui
  gui/OpenSceneDialog.ui
  gui/RenderParamsWindow.ui
)

# Find includes in corresponding build directories.
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)

# Create UI header files.
QT5_WRAP_UI(UIS_H ${UIS})

source_group("Resource Files" FILES MainWindow.qrc)
source_group("Generated Files" FILES ${MOC_SRCS} ${UIS_H})
source_group("Form Files" FILES ${UIS})

source_group("Header Files\\" FILES ${ROOT_HDR})
source_group("Header Files\\benchmark" FILES ${BENCHMARK_HDR})
source_group("Header Files\\environment" FILES ${ENVIRONMENT_HDR})
source_group("Header Files\\gpu" FILES ${GPU_HDR})
source_group("Header Files\\gui" FILES ${GUI_HDR})
source_group("Header Files\\radix_sort" FILES ${RADIX_SORT_HDR})
source_group("Header Files\\random" FILES ${RANDOM_HDR})
source_group("Header Files\\rt" FILES ${RT_HDR})
source_group("Header Files\\rt\\bvh" FILES ${RT_BVH_HDR})
source_group("Header Files\\rt\\ray" FILES ${RT_RAY_HDR})
source_group("Header Files\\rt\\renderer" FILES ${RT_RENDERER_HDR})
source_group("Header Files\\rt\\scene" FILES ${RT_SCENE_HDR})
source_group("Header Files\\rt\\tracer" FILES ${RT_TRACER_HDR})
source_group("Header Files\\shader" FILES ${SHADER_HDR})
source_group("Header Files\\util" FILES ${UTIL_HDR})

source_group("Source Files\\" FILES ${ROOT_SRC})
source_group("Source Files\\benchmark" FILES ${BENCHMARK_SRC})
source_group("Source Files\\environment" FILES ${ENVIRONMENT_SRC})
source_group("Source Files\\gpu" FILES ${GPU_SRC})
source_group("Source Files\\gui" FILES ${GUI_SRC})
source_group("Source Files\\radix_sort" FILES ${RADIX_SORT_SRC})
source_group("Source Files\\random" FILES ${RANDOM_SRC})
source_group("Source Files\\rt\\bvh" FILES ${RT_BVH_SRC})
source_group("Source Files\\rt\\ray" FILES ${RT_RAY_SRC})
source_group("Source Files\\rt\\renderer" FILES ${RT_RENDERER_SRC})
source_group("Source Files\\rt\\scene" FILES ${RT_SCENE_SRC})
source_group("Source Files\\rt\\tracer" FILES ${RT_TRACER_SRC})
source_group("Source Files\\shader" FILES ${SHADER_SRC})
source_group("Source Files\\util" FILES ${UTIL_SRC})

# Include UI dir.
set(UI_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/gui)

# Project name.
set(NAME hippie)

# Add all files to excutable.
add_executable(${NAME}
  ${ROOT_HDR}
  ${ROOT_SRC}
  ${BENCHMARK_HDR}
  ${BENCHMARK_SRC}
  ${ENVIRONMENT_HDR}
  ${ENVIRONMENT_SRC}
  ${GPU_HDR}
  ${GPU_SRC}
  ${GUI_HDR}
  ${GUI_SRC}
  ${RADIX_SORT_HDR}
  ${RADIX_SORT_SRC}
  ${RANDOM_HDR}
  ${RANDOM_SRC}
  ${RT_HDR}
  ${RT_BVH_HDR}
  ${RT_BVH_SRC}
  ${RT_RAY_HDR}
  ${RT_RAY_SRC}
  ${RT_RENDERER_HDR}
  ${RT_RENDERER_SRC}
  ${RT_SCENE_HDR}
  ${RT_SCENE_SRC}
  ${RT_TRACER_HDR}
  ${RT_TRACER_SRC}
  ${SHADER_SRC}
  ${UTIL_HDR}
  ${UTIL_SRC}
  ${UIS_H} 
  ${MOC_SRCS}
)

if(${CUDA_USE_CUDA})
# Add include directories.
include_directories (SYSTEM
  ${ASSIMP_INCLUDE_DIR}
  ${DEVIL_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${HIP_INCLUDE_DIR}
  ${GLEW_INCLUDE_DIR}
  ${OPENGL_INCLUDE_DIR}
  ${UI_INCLUDE_DIR}
)
target_link_libraries (${NAME}
  ${ASSIMP_LIBRARIES}
  ${DEVIL_LIBRARIES}
  ${CUDA_DRIVER_API_LIBRARY}
  ${CUDA_RUNTIME_API_LIBRARY}
  ${CUDA_NVRTC_LIBRARY}
  ${GLEW_LIBRARIES}
  ${OPENGL_LIBRARIES}
  Qt5::Core
  Qt5::Widgets
  Qt5::OpenGL
)
else(${CUDA_USE_CUDA})
include_directories (SYSTEM
  ${ASSIMP_INCLUDE_DIR}
  ${DEVIL_INCLUDE_DIR}
  ${HIP_INCLUDE_DIR}
  ${GLEW_INCLUDE_DIR}
  ${OPENGL_INCLUDE_DIR}
  ${UI_INCLUDE_DIR}
)
target_link_libraries (${NAME}
  ${ASSIMP_LIBRARIES}
  ${DEVIL_LIBRARIES}
  ${HIP_DRIVER_API_LIBRARY}
  ${GLEW_LIBRARIES}
  ${OPENGL_LIBRARIES}
  Qt5::Core
  Qt5::Widgets
  Qt5::OpenGL
)
endif(${CUDA_USE_CUDA})

# Set output directory.
set(BIN_DIR ${CMAKE_SOURCE_DIR}/bin)

# Properties.
set_target_properties(${NAME} PROPERTIES 
	DEBUG_OUTPUT_NAME ${NAME}_d 
	RELEASE_OUTPUT_NAME ${NAME}
    RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR}
	RUNTIME_OUTPUT_DIRECTORY_RELEASE ${BIN_DIR}
	RUNTIME_OUTPUT_DIRECTORY_DEBUG ${BIN_DIR}
)

if (WIN32)
  set_target_properties(${NAME} PROPERTIES 
    LINK_FLAGS "/NODEFAULTLIB:LIBCMTD /NODEFAULTLIB:LIBCMT /NODEFAULTLIB:LIBC"
  )
endif (WIN32)
