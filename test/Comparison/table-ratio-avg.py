import numpy as np

benchmark_dir = "../../bin/benchmark/"

mode = 1
# 0 perf
# 1 relative time

scenes = ["conference", "buddha", "sodahall", "hairball", "manuscript", "crown", "Pompeii", "san-miguel", "Vienna", "powerplant", "sponza", "crytek-sponza", "sibenik", "gallery"]
scene_indices = [12, 11, 0, 13, 1, 2, 3, 5, 6, 7, 8, 9]
number_of_scenes = len(scene_indices)

sorts = [False, True]
layouts = ["bin", "quad", "oct"]
layout_labels = ["Binary", "Quaternary", "Octal"]
layout_labels = ["2", "4", "8"]
cameras = [0, 1, 2]
number_of_cameras = 3

ray_types = ["PATH", "PATH", "SHADOW", "PATH", "SHADOW"]
sorts = [False, False, False, True, True]
stat_labels = ["BVH cost", "Second. rays", "Shadow rays", "Second. rays reord.", "Shadow rays reord."]

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

table_file = "table-ratio-avg.tex"


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


def parse_stat(method, scene, splitting, annealing, sort, stat, layout, camera):
    test_name = scene + "-" + layout + "-" + method
    test_name += "-split" + str(int(splitting))
    test_name += "-anneal" + str(int(annealing))
    test_name += "-sort" + str(int(sort))
    test_name += "-cam" + str(int(camera))
    test_name += "-ct" + str(ct)
    test_name += "-ci" + str(ci)
    log_file = benchmark_dir + test_name + "/test_" + test_name + ".log"
    print(log_file)
    value = parse_stat_from_log(log_file, stat)
    return value


table = open(table_file, "w")
for layout_index in range(len(layouts)):
    layout = layouts[layout_index]
    layout_label = layout_labels[layout_index]
    ref_layout = "bin"
    if mode >= 1:
        ref_layout = layout
    table.write("\\hline\n")
    table.write(layout_label)
    for stat_index in range(len(stat_labels)):
        if mode == 0 and stat_index == 0:
            continue
        stat_label = stat_labels[stat_index]
        sort = sorts[stat_index]
        ray_type = ray_types[stat_index]
        num_rays = "NUMBER OF " + ray_type
        stat = "RT PERFORMANCE " + ray_type
        table.write(" & " + stat_label)
        g_perfs = []
        g_ratios = []
        g_costs = []
        for method_index in range(len(methods)):
            method = methods[method_index]
            annealing = annealings[method_index]
            splitting = splittings[method_index]
            perfs = []
            ratios = []
            costs = []
            for scene_index in scene_indices:
                scene = scenes[scene_index]
                for camera in cameras:
                    cost_ref = parse_stat("lbvh", scene, False, False, False, "BVH COST", ref_layout, camera)
                    cost = parse_stat(method, scene, splitting, annealing, sort, "BVH COST", layout, camera)
                    costs.append(cost / cost_ref)
                    rays_ref = parse_stat("lbvh", scene, False, False, False, num_rays, ref_layout, camera)
                    rays = parse_stat(method, scene, splitting, annealing, sort, num_rays, layout, camera)
                    perf_ref = parse_stat("lbvh", scene, False, False, False, stat, ref_layout, camera)
                    perf = parse_stat(method, scene, splitting, annealing, sort, stat, layout, camera)
                    perfs.append(perf / perf_ref)
                    time = rays * 1.0e-6 / perf
                    time_ref = rays_ref * 1.0e-6 / perf_ref
                    ratio = time / cost
                    ratio_ref = time_ref / cost_ref
                    ratios.append(ratio / ratio_ref)
            avg_perf = np.array(perfs).mean()
            avg_ratio = np.array(ratios).mean()
            avg_cost = np.array(costs).mean()
            g_perfs.append(avg_perf)
            g_ratios.append(avg_ratio)
            g_costs.append(avg_cost)
        g_costs = np.array(g_costs)
        g_perfs = np.array(g_perfs)
        g_ratios = np.array(g_ratios)
        best_val = np.max(g_perfs)
        for method_index in range(len(methods)):
            if mode == 1:
                value = g_ratios[method_index]
                if stat_index == 0:
                    value = g_costs[method_index]
                table.write(" & " + (precision2 % value))
            else:
                value = g_perfs[method_index]
                if value == best_val:
                    table.write(" & \\textbf{" + (precision2 % value) + "}")
                else:
                    table.write(" & " + (precision2 % value))
        table.write("\\\\\n")


table.close()
