import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy import stats
from math import log

benchmark_dir = "../../bin/benchmark/"

scenes = ["conference", "buddha", "sodahall", "hairball", "manuscript", "crown", "Pompeii", "san-miguel", "Vienna", "powerplant", "sponza", "crytek-sponza", "sibenik", "gallery"]
scene_indices = [12, 11, 0, 13, 1, 2, 3, 5, 6, 7, 8, 9]

sorts = [False, True]
layout = "bin"
cameras = [0, 1, 2]
methods = ["lbvh", "hlbvh", "ploc", "atr", "insertion", "insertion", "sbvh", "sbvh"]
method_labels = ["LBVH", "HLBVH", "PLOC", "ATRBVH", "PRBVH", "PRBVH$_S^A$", "SAH", "SBVH"]
annealings = [False, False, False, False, False, True, False, False]
splittings = [False, False, False, False, False, True, False, True]
number_of_methods = len(methods)

ray_types = [["PATH", "path", "Secondary rays"], ["SHADOW", "shadow", "Shadow rays"]]
ct = 3
ci = 2


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
        test_name += "-ct" + str(ct)
        test_name += "-ci" + str(ci)
        log_file = benchmark_dir + test_name + "/test_" + test_name + ".log"
        # print(log_file)
        value += parse_stat_from_log(log_file, stat) / len(cameras)
    return value


colors = cm.rainbow(np.linspace(0, 1, number_of_methods))
for sort in sorts:
    for ray_type in ray_types:
        Xs = np.array([])
        Ys = np.array([])
        fig, ax = plt.subplots()
        for method_index in range(number_of_methods):
            method = methods[method_index]
            method_label = method_labels[method_index]
            annealing = annealings[method_index]
            splitting = splittings[method_index]
            X = []
            Y = []
            color = colors[method_index]
            for scene_index in scene_indices:
                scene = scenes[scene_index]
                cost_ref = parse_stat("lbvh", scene, False, False, sort, "BVH COST", layout)
                cost = parse_stat(method, scene, splitting, annealing, sort, "BVH COST", layout)
                rays_ref = parse_stat("lbvh", scene, False, False, sort, "NUMBER OF " + ray_type[0] + " RAYS", layout)
                rays = parse_stat(method, scene, splitting, annealing, sort, "NUMBER OF " + ray_type[0] + " RAYS", layout)
                perf_ref = parse_stat("lbvh", scene, False, False, sort, "RT PERFORMANCE " + ray_type[0], layout)
                perf = parse_stat(method, scene, splitting, annealing, sort, "RT PERFORMANCE " + ray_type[0], layout)
                time = rays * 1.0e-6 / perf
                time_ref = rays_ref * 1.0e-6 / perf_ref
                X.append((cost / cost_ref))
                Y.append((time / time_ref))
            print(min(X))
            if method_index != -1:
                res = stats.linregress(X, Y)
                plt.plot(X, [res.intercept + res.slope*x for x in X], c=color)
            Xs = np.append(Xs,X)
            Ys = np.append(Ys,Y)
            print(X,Y)
            ax.scatter(X, Y, c=color, label=method_label, edgecolors='none')
        ax.legend(loc='lower right', ncol=2)
        plt.xlabel("norm. BVH cost")
        plt.ylabel("norm. trace time")
        plt.title(ray_type[2] + (" with reordering" if sort else ""))
        ax.grid(True)

        print(min(Xs), min(Ys))
        print(sorted(Xs))

        print(Xs,Ys)
        res = stats.linregress(Xs, Ys)
        # print(res)
        # print(Xs, res.intercept + res.slope*Xs)
        plt.plot(Xs, res.intercept + res.slope*Xs, linewidth=2, label='fitted line')
        print('pearson = ', stats.pearsonr(Xs,Ys)[0])
        print('spearman = ', stats.spearmanr(Xs,Ys).correlation)
        print('kendall = ', stats.kendalltau(Xs,Ys).correlation)

        # plt.show()
        plt.savefig("bvh_corr_" + ray_type[1] + ("_sort" if sort else "") +".pdf")
