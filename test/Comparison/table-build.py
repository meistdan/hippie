import numpy as np

benchmark_dir = "../../bin/benchmark/"

scenes = ["conference", "buddha", "sodahall", "hairball", "manuscript", "crown", "Pompeii", "san-miguel", "Vienna", "powerplant", "sponza", "crytek-sponza", "sibenik", "gallery"]
scene_labels = ["Conference", "Happy Buddha", "Sodahall", "Hairball", "Manuscript", "Crown", "Pompeii", "San Miguel", "Vienna", "Powerplant", "Sponza", "Crytek Sponza", "Sibenik", "Gallery"]
scene_indices = [12, 11, 0, 13, 1, 2, 3, 5, 6, 7, 8, 9]
tris_nums = ["331k", "1087k", "2169k", "2880k", "4305k", "4868k", "5632k", "7880k", "8637k", "12759k", "66k", "262k", "75k", "998k"]

sort = False
layout = "bin"
cameras = [0, 1, 2]

methods = ["lbvh", "hlbvh", "ploc", "atr", "insertion", "insertion", "insertion", "insertion", "sbvh", "sbvh"]
annealings = [False, False, False, False, False, True, False, True, False, False]
splittings = [False, False, False, False, False, False, True, True, False, True]

precision0 = "%.0f"
precision2 = "%.2f"
ct = 3
ci = 2


def format(value):
    if value < 100:
        return "%.1f" % value
    return str(int(value))

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
        print(log_file)
        value += parse_stat_from_log(log_file, stat) / len(cameras)
    return value


table_file = "table-build-" + layout + ".tex"
table = open(table_file, "w")
data_avg = np.zeros(len(methods))
table.write("\\hline\n")
for scene_index in scene_indices:
    scene = scenes[scene_index]
    scene_label = scene_labels[scene_index]

    data = []
    data_rel = []
    table.write(scene_label)
    for method_index in range(len(methods)):
        method = methods[method_index]
        annealing = annealings[method_index]
        splitting = splittings[method_index]
        value_ref = parse_stat("lbvh", scene, False, False, sort, "BVH CONSTRUCTION TIME KERNELS", layout)
        value = parse_stat(method, scene, splitting, annealing, sort, "BVH CONSTRUCTION TIME KERNELS", layout)
        if method_index == 6 or method_index == 7:
            sbvh_method = methods[9]
            sbvh_annealing = annealings[9]
            sbvh_splitting = splittings[9]
            value += parse_stat(sbvh_method, scene, sbvh_splitting, sbvh_annealing, sort, "BVH CONSTRUCTION TIME ABSOLUTE", layout)
        if value == 0:
            value = parse_stat(method, scene, splitting, annealing, sort, "BVH CONSTRUCTION TIME ABSOLUTE", layout)
        data.append(value * 1000)
        data_rel.append(value / value_ref)
        data_avg[method_index] += data_rel[-1] / len(scene_indices)
    data = np.array(data)
    best_val = np.min(data)
    for method_index in range(len(methods)):
        value = data[method_index]
        value_rel = data_rel[method_index]
        # if data[method_index] == best_val:
        #     table.write(" & \\textbf{" + format(value) + "}")
        # else:
        #     table.write(" & " + format(value))
        table.write(" & " + format(value))
    table.write("\\\\\n")

table.write("\\hline\n")
table.write("Avg. build time")
best_val = np.min(data_avg)
for method_index in range(len(methods)):
    value = data_avg[method_index]
    # if data_avg[method_index] == best_val:
    #     table.write(" & \\textbf{" + format(value) + "}")
    # else:
    #     table.write(" & " + format(value))
    table.write(" & " + format(value))
table.write("\\\\\n")
table.close()
