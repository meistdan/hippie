# Hippie
This framework was originally created for my masters thesis. I have been extending and maintaining it thourgh years. It served as as common code base for my research during my PhD studies at the Czech Technical University in Prague. The framework contains a collection BVH construction algorithms and other related techniques implemented in HIP.

## Requirements
The framework was recently tested only on Windows but originally it runs also on Linux.
- GPU: AMD Navi1x+ or Nvidia Kepler+
- ROCm 4.4+ (Driver 21.50+) or CUDA 11.4+
- CMake 2.8.10+
- QT 5.1+ (OpenGL version! Not ANGLE version!)
- GLEW 1.8+
- Assimp 3.1.1+
- DevIL 1.7.8+
- Windows:  Visual Studio 2019+ (x64)
- Linux: g++ 4.8.1+

Note that the code assumes warp size 32, so it most likely won't work on the Vega architecture.

## Compilation
Windows:
- Set environment variable GLEW as GLEW root directory.
- Set environment variable ASSIMP as Assimp root directory.
- Set environment variable QT5_DIR as QT root directory.
- If you want to use HIP, set environment variable HIP_DIR as HIP root directory.
- Run CMake gui.
- Set source code as the root directory.
- Set build as the root/build directory.
- Configure and choose Visual Studio solution (x64).
- If you want to use CUDA, check the CUDA_USE_CUDA option.
- Genereate a solution.
- Build the solution.
 
Linux:
- If you want to use HIP, set environment variable HIP_DIR as HIP root directory.
- Run CMake gui or CMake.
- Set source code as the trunk directory.
- Set build as the root/build directory.
- Configure and choose makefile.
- If you want to use CUDA, check the CUDA_USE_CUDA option.
- Generate makefile.
- Go to build directory and run make.

Note: The executable must resides in the root/bin because of paths.

## Usage
We use env file format for the configuration, which is a simple text based format using keys and values. All options can be found in src/environment/AppEnvironment.cpp. We simply use the env file as argument to run the sample:
```
./hippie.exe config.env
```
If we do not use any file, then bin/default.env is used instead.

## References
The framework is based on <a href="https://code.google.com/archive/p/understanding-the-efficiency-of-ray-traversal-on-gpus/">Understanding the Efficiency of Ray Traversal on GPUs</a>. 

Besides that, we use modified versions of the following public implementations:
- <a href="https://github.com/leonardo-domingues/atrbvh/">Bounding Volume Hierarchy Optimization through Agglomerative Treelet Restructuring</a>
- <a href="https://github.com/lispbub/simd-ray-traversal/">CPU-Style SIMD Ray Traversal on GPUs</a>
- <a href="https://nvlabs.github.io/cub/">High-Performance and Scalable Radix Sorting: A case study of implementing dynamic parallelism for GPU computing</a>

## Citation
If you use this code, please cite <a href="https://jcgt.org/published/0011/02/08/">the paper</a>:
```
@Article{Meister2022,
  author = {Daniel Meister and Ji\v{r}\'{\i} Bittner},
  title = {{Performance Comparison of Bounding Volume Hierarchies for GPU Ray Tracing}},
  journal = {Journal of Computer Graphics Techniques},
  volume = {11},
  number = {2},
  pages = {143--161},
  year = {2022},
}
```

## Disclaimer
I implemented this work during my stays at the Czech Technical University and the University of Tokyo before I joined AMD.
