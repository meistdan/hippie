import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy import stats

benchmark_dir = "../../bin/benchmark/"

scenes = ["conference", "buddha", "sodahall", "hairball", "manuscript", "crown", "Pompeii", "san-miguel", "Vienna", "powerplant", "sponza", "crytek-sponza", "sibenik"]
scene_indices = [10, 12, 11, 0, 1, 2, 3, 5, 6, 7, 8, 9]

sort = False
layout = "bin"
cameras = [0, 1, 2]
methods = ["lbvh", "hlbvh", "ploc", "atr", "insertion", "insertion", "sbvh", "sbvh"]
method_labels = ["LBVH", "HLBVH", "PLOC", "ATRBVH", "PRBVH", "PRBVHs", "SAH", "SBVH"]
annealings = [False, False, False, False, False, True, False, False]
splittings = [False, False, False, False, False, True, False, True]
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
        # print(log_file)
        value += parse_stat_from_log(log_file, stat) / len(cameras)
    return value


Xs = []
fig, ax = plt.subplots()
colors = cm.rainbow(np.linspace(0, 1, number_of_methods))
for method_index in range(number_of_methods):
    method = methods[method_index]
    method_label = method_labels[method_index]
    annealing = annealings[method_index]
    splitting = splittings[method_index]
    X = []
    Y = []
    D = []
    color = colors[method_index]
    for scene_index in scene_indices:
        scene = scenes[scene_index]
        cost_ref = parse_stat("lbvh", scene, False, False, False, "BVH COST", layout)
        cost = parse_stat(method, scene, splitting, annealing, sort, "BVH COST", layout)
        rays_ref = parse_stat("lbvh", scene, False, False, False, "NUMBER OF PATH", layout)
        rays = parse_stat(method, scene, splitting, annealing, sort, "NUMBER OF PATH", layout)
        perf_ref = parse_stat("lbvh", scene, False, False, False, "RT PERFORMANCE PATH", layout)
        perf = parse_stat(method, scene, splitting, annealing, sort, "RT PERFORMANCE PATH", layout)
        time = rays * 1.0e-6 / perf
        time_ref = rays_ref * 1.0e-6 / perf_ref
        # X.append(cost / cost_ref)
        # Y.append(time / time_ref)
        X.append(cost)
        Y.append(time)
        D.append((time / cost) / (time_ref / cost_ref))
    # D = [Y[i]/X[i] for i in range(len(X))]
    ax.scatter([method_index for x in D], D, c=color, label=method_label, edgecolors='none')
    print(D)
    if len(Xs) == 0: 
        Xs = np.array(D)
    else:
        Xs = np.vstack([Xs, D])
print(Xs)
ax.legend(loc='upper right', ncol=4)
plt.draw()

print('Homogeneity of variance Assumption Check = ', Xs[1:].std(axis=1).max() / Xs[1:].std(axis=1).min(), ', should be < 2.0')

print('Normality Assumption Check')
fig = plt.figure()
for method_index in range(number_of_methods):
    ax = fig.add_subplot(3,3,method_index+1)
    stats.probplot(Xs[method_index], dist="norm", plot=ax)
    ax.set_title("Probability Plot - " +  methods[method_index])

plt.show()

print("ANOVA for all: ", stats.f_oneway(Xs[0],Xs[1],Xs[2],Xs[3],Xs[4],Xs[5],Xs[6],Xs[7]))

# now we perform paired t-tests
# no need to check for homogeneity of variance because we test pairs
print('Paired t-tests: ')
GENERATE_GRAPHVIZ = True
for method_index_0 in range(number_of_methods):
    for method_index_1 in range(method_index_0+1, number_of_methods):
        if GENERATE_GRAPHVIZ:
            print(method_labels[method_index_0], ' -- ', method_labels[method_index_1], ' [ len =',round(10.0*(1.0 - stats.ttest_rel(Xs[method_index_0],Xs[method_index_1]).pvalue),2),']')
        else:
            print(method_labels[method_index_0], method_labels[method_index_1], stats.ttest_rel(Xs[method_index_0],Xs[method_index_1]).pvalue)
        # if Xs[[method_index_0,method_index_1],:].std(axis=1).max() / Xs[[method_index_0,method_index_1],:].std(axis=1).min() < 3.0 or True:
            # print(method_labels[method_index_0], ' -- ', method_labels[method_index_1], ' [len=',round(10*(1 - stats.kruskal(Xs[method_index_0],Xs[method_index_1]).pvalue),2),']')
        # print(method_labels[method_index_0], method_labels[method_index_1], ', homogeneity of variance Assumption Check = ', Xs[[method_index_0,method_index_1],:].std(axis=1).max() / Xs[[method_index_0,method_index_1],:].std(axis=1).min(), ', should be < 2.0, stats = ', stats.f_oneway(Xs[method_index_0],Xs[method_index_1]))
