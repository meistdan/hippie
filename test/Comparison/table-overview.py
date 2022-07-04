import numpy as np

benchmark_dir = "../../bin/benchmark/"

scenes = ["conference", "buddha", "sodahall", "hairball", "manuscript", "crown", "Pompeii", "san-miguel", "Vienna", "powerplant", "sponza", "crytek-sponza", "sibenik", "gallery"]
scene_labels = ["Conference", "Happy Buddha", "Sodahall", "Hairball", "Manuscript", "Crown", "Pompeii", "San Miguel", "Vienna", "Powerplant", "Sponza", "Crytek Sponza", "Sibenik", "Gallery"]
scene_indices = [12, 11, 0, 13, 1, 2, 3, 5, 6, 7, 8, 9]
tris_nums = ["331k", "1087k", "2169k", "2880k", "4305k", "4868k", "5632k", "7880k", "8637k", "12759k", "66k", "262k", "75k", "998k"]

sorts = [False, True]
layouts = ["bin", "quad", "oct"]
cameras = [0, 1, 2]

# methods = ["lbvh", "hlbvh", "ploc", "atr", "insertion", "insertion", "sbvh", "sbvh"]
# annealings = [False, False, False, False, False, True, False, False]
# splittings = [False, False, False, False, False, True, False, True]

methods = ["lbvh", "hlbvh", "ploc", "atr", "insertion", "insertion", "insertion", "insertion", "sbvh", "sbvh"]
annealings = [False, False, False, False, False, True, False, True, False, False]
splittings = [False, False, False, False, False, False, True, True, False, True]

precision0 = "%.0f"
precision2 = "%.2f"
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
        print(log_file)
        value += parse_stat_from_log(log_file, stat) / len(cameras)
    return value


def run(scene_index, layout):

    scene = scenes[scene_index]
    scene_label = scene_labels[scene_index]
    tris_num = tris_nums[scene_index]

    table.write("\\hline\n")

    data = []
    data_rel = []
    table.write(" & BVH cost")
    for method_index in range(len(methods)):
        method = methods[method_index]
        annealing = annealings[method_index]
        splitting = splittings[method_index]
        value_ref = 0.5 * (parse_stat("lbvh", scene, False, False, True, "BVH COST", layout) + parse_stat("lbvh", scene, False, False, False, "BVH COST", layout))
        value = 0.5 * (parse_stat(method, scene, splitting, annealing, True, "BVH COST", layout) + parse_stat(method, scene, splitting, annealing, False, "BVH COST", layout))
        data.append(value)
        data_rel.append(value / value_ref)
    data = np.array(data)
    best_val = np.min(data)
    for method_index in range(len(methods)):
        value = data[method_index]
        value_rel = data_rel[method_index]
        if data[method_index] == best_val:
            table.write(" & \\textbf{" + (precision0 % value) + " (" + (precision2 % value_rel) + ")}")
        else:
            table.write(" & " + (precision0 % value) + " (" + (precision2 % value_rel) + ")")
    table.write("\\\\\n")

    data = []
    data_rel = []
    table.write(scene_label)
    table.write(" & Second. rays")
    for method_index in range(len(methods)):
        method = methods[method_index]
        annealing = annealings[method_index]
        splitting = splittings[method_index]
        value_ref = parse_stat("lbvh", scene, False, False, False, "RT PERFORMANCE PATH", layout)
        value = parse_stat(method, scene, splitting, annealing, False, "RT PERFORMANCE PATH", layout)
        data.append(value)
        data_rel.append(value / value_ref)
    data = np.array(data)
    best_val = np.max(data)
    for method_index in range(len(methods)):
        value = data[method_index]
        value_rel = data_rel[method_index]
        if data[method_index] == best_val:
            table.write(" & \\textbf{" + (precision0 % value) + " (" + (precision2 % value_rel) + ")}")
        else:
            table.write(" & " + (precision0 % value) + " (" + (precision2 % value_rel) + ")")
    table.write("\\\\\n")

    data = []
    data_rel = []
    table.write("\\#triangles")
    table.write(" & Shadow rays")
    for method_index in range(len(methods)):
        method = methods[method_index]
        annealing = annealings[method_index]
        splitting = splittings[method_index]
        value_ref = parse_stat("lbvh", scene, False, False, False, "RT PERFORMANCE SHADOW", layout)
        value = parse_stat(method, scene, splitting, annealing, False, "RT PERFORMANCE SHADOW", layout)
        data.append(value)
        data_rel.append(value / value_ref)
    data = np.array(data)
    best_val = np.max(data)
    for method_index in range(len(methods)):
        value = data[method_index]
        value_rel = data_rel[method_index]
        if data[method_index] == best_val:
            table.write(" & \\textbf{" + (precision0 % value) + " (" + (precision2 % value_rel) + ")}")
        else:
            table.write(" & " + (precision0 % value) + " (" + (precision2 % value_rel) + ")")
    table.write("\\\\\n")

    data = []
    data_rel = []
    table.write(tris_num)
    table.write(" & Second. rays reord.")
    for method_index in range(len(methods)):
        method = methods[method_index]
        annealing = annealings[method_index]
        splitting = splittings[method_index]
        value_ref = parse_stat("lbvh", scene, False, False, False, "RT PERFORMANCE PATH", layout)
        value = parse_stat(method, scene, splitting, annealing, True, "RT PERFORMANCE PATH", layout)
        data.append(value)
        data_rel.append(value / value_ref)
    data = np.array(data)
    best_val = np.max(data)
    for method_index in range(len(methods)):
        value = data[method_index]
        value_rel = data_rel[method_index]
        if data[method_index] == best_val:
            table.write(" & \\textbf{" + (precision0 % value) + " (" + (precision2 % value_rel) + ")}")
        else:
            table.write(" & " + (precision0 % value) + " (" + (precision2 % value_rel) + ")")
    table.write("\\\\\n")

    data = []
    data_rel = []
    table.write(" & Shadow rays reord.")
    for method_index in range(len(methods)):
        method = methods[method_index]
        annealing = annealings[method_index]
        splitting = splittings[method_index]
        value_ref = parse_stat("lbvh", scene, False, False, False, "RT PERFORMANCE SHADOW", layout)
        value = parse_stat(method, scene, splitting, annealing, True, "RT PERFORMANCE SHADOW", layout)
        data.append(value)
        data_rel.append(value / value_ref)
    data = np.array(data)
    best_val = np.max(data)
    for method_index in range(len(methods)):
        value = data[method_index]
        value_rel = data_rel[method_index]
        if data[method_index] == best_val:
            table.write(" & \\textbf{" + (precision0 % value) + " (" + (precision2 % value_rel) + ")}")
        else:
            table.write(" & " + (precision0 % value) + " (" + (precision2 % value_rel) + ")")
    table.write("\\\\\n")


for layout in layouts:
    table_file = "table-overview-" + layout + ".tex"
    table = open(table_file, "w")
    for scene_index in scene_indices:
        run(scene_index, layout)
    table.close()
