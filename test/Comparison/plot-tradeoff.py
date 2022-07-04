import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

benchmark_dir = "../../bin/benchmark/"

scenes = ["conference", "buddha", "sodahall", "hairball", "manuscript", "crown", "Pompeii", "san-miguel", "Vienna", "powerplant", "sponza", "crytek-sponza", "sibenik"]
scene_indices = [10, 12, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

sort = False
layout = "bin"
cameras = [0, 1, 2]
methods = ["lbvh", "hlbvh", "ploc", "atr", "insertion"]
method_labels = ["LBVH", "HLBVH", "PLOC", "ATRBVH", "PRBVH"]
annealings = [False, False, False, False, False]
splittings = [False, False, False, False, False]
number_of_methods = len(methods)


def parse_stat_from_log(log_file, stat):
    file = open(log_file)
    for line in file:
        if stat in line:
            res = float(file.readline())
            file.close()
            return res
    file.close()
    print("stat was not found: " + stat)
    return 0


def parse_stat(method, scene, splitting, annealing, sort, stat, layout):
    value = 0
    for camera in cameras:
        test_name = scene + "-" + layout + "-" + method
        test_name += "-split" + str(int(splitting))
        test_name += "-anneal" + str(int(annealing))
        test_name += "-sort" + str(int(sort))
        test_name += "-cam" + str(int(camera))
        log_file = benchmark_dir + test_name + "/test_" + test_name + ".log"
        print(log_file)
        value += parse_stat_from_log(log_file, stat) / len(cameras)
    return value


fig, ax = plt.subplots()
colors = cm.rainbow(np.linspace(0, 1, number_of_methods))
for method_index in range(number_of_methods):
    method = methods[method_index]
    method_label = method_labels[method_index]
    annealing = annealings[method_index]
    splitting = splittings[method_index]
    X = []
    Y = []
    color = colors[method_index]
    avg_value = 0
    for scene_index in scene_indices:
        scene = scenes[scene_index]
        build_time = parse_stat(method, scene, splitting, annealing, sort, "BVH CONSTRUCTION TIME KERNELS", layout) * 1.0e3
        build_time_ref = parse_stat("lbvh", scene, False, False, False, "BVH CONSTRUCTION TIME KERNELS", layout) * 1.0e3
        rays = parse_stat(method, scene, splitting, annealing, sort, "NUMBER OF PATH", layout)
        rays_ref = parse_stat("lbvh", scene, False, False, False, "NUMBER OF PATH", layout)
        perf = parse_stat(method, scene, splitting, annealing, sort, "RT PERFORMANCE PATH", layout)
        perf_ref = parse_stat("lbvh", scene, False, False, False, "RT PERFORMANCE PATH", layout)
        trace_time = rays / perf * 1.0e-3
        trace_time_ref = rays_ref / perf_ref * 1.0e-3
        X.append(build_time / build_time_ref)
        Y.append(trace_time / trace_time_ref)
    x = np.array(X).mean()
    y = np.array(Y).mean()
    ax.scatter(x, y, c=color, label=method_label, edgecolors='none')
ax.legend()
plt.xlabel("build time [ms]")
plt.ylabel("trace time [ms]")
ax.grid(True)

plt.show()
