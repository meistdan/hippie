import numpy as np

benchmark_dir = "../../bin/benchmark/"

mode = 0
# 0 perf
# 1 time-tom-image
# 2 relative time

scenes = ["conference", "buddha", "sodahall", "hairball", "manuscript", "crown", "Pompeii", "san-miguel", "Vienna", "powerplant", "sponza", "crytek-sponza", "sibenik"]
scene_indices = [10, 12, 11, 0, 1, 2, 3, 5, 6, 7, 8, 9]
number_of_scenes = len(scene_indices)

sorts = [False, True]
layouts = ["bin", "quad", "oct"]
layout_labels = ["Binary", "Quaternary", "Octal"]
layout_labels = ["2", "4", "8"]
# layouts = ["quad", "oct"]
# layout_labels = ["Quaternary", "Octal"]
# layout_labels = ["4", "8"]
cameras = [0, 1, 2]
number_of_cameras = 3
cs = [[1, 8], [1, 4], [1, 2], [1, 1], [3, 2], [2, 1], [5, 2]]
# cs = [[1, 8]]

ray_types = ["PATH", "PATH", "PATH", "SHADOW", "SHADOW"]
sorts = [False, False, True, False, True]
stat_labels = ["Second. rays", "Second. rays sort.", "Shadow rays", "Shadow rays sort."]

methods = ["lbvh", "hlbvh", "ploc", "atr", "sbvh"]
annealings = [False, False, False, False, False]
splittings = [False, False, False, False, False]

precision0 = "%.0f"
precision2 = "%.2f"

table_file = "table-constants-avg.tex"


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


def parse_stat(method, scene, splitting, annealing, sort, stat, layout, camera, ct, ci):
    test_name = scene + "-" + layout + "-" + method
    test_name += "-split" + str(int(splitting))
    test_name += "-anneal" + str(int(annealing))
    test_name += "-sort" + str(int(sort))
    test_name += "-cam" + str(int(camera))
    test_name += "-ct" + str(ct)
    test_name += "-ci" + str(ci)
    # if layout != "bin":
    #     test_name += "-sl"
    log_file = benchmark_dir + test_name + "/test_" + test_name + ".log"
    print(log_file)
    value = parse_stat_from_log(log_file, stat)
    return value


table = open(table_file, "w")
for layout_index in range(len(layouts)):
    layout = layouts[layout_index]
    layout_label = layout_labels[layout_index]
    ref_layout = "bin"
    if mode >= 2:
        ref_layout = layout
    table.write("\\hline\n")
    table.write(layout_label)
    for stat_index in range(len(stat_labels)):
        stat_label = stat_labels[stat_index]
        sort = sorts[stat_index]
        ray_type = ray_types[stat_index]
        num_rays = "NUMBER OF " + ray_type
        stat = "RT PERFORMANCE " + ray_type
        for c in cs:
            ct = c[0]
            ci = c[1]
            ref_ct = 3
            ref_ci = 2
            table.write(" & " + stat_label + " " + str(ct / ci))
            g_perfs = []
            g_times = []
            g_ratios = []
            g_ttis = []
            for method_index in range(len(methods)):
                method = methods[method_index]
                annealing = annealings[method_index]
                splitting = splittings[method_index]
                times = []
                perfs = []
                ratios = []
                ttis = []
                for scene_index in scene_indices:
                    scene = scenes[scene_index]
                    for camera in cameras:
                        tti_ref = parse_stat("lbvh", scene, False, False, False, "TIME TO IMAGE KERNELS", ref_layout, camera, ref_ct, ref_ci)
                        tti = parse_stat(method, scene, splitting, annealing, sort, "TIME TO IMAGE KERNELS", layout, camera, ct, ci)
                        ttis.append(tti / tti_ref)
                        cost_ref = parse_stat("lbvh", scene, False, False, False, "BVH COST", ref_layout, camera, ref_ct, ref_ci)
                        cost = parse_stat(method, scene, splitting, annealing, sort, "BVH COST", layout, camera, ct, ci)
                        rays_ref = parse_stat("lbvh", scene, False, False, False, num_rays, ref_layout, camera, ref_ct, ref_ci)
                        rays = parse_stat(method, scene, splitting, annealing, sort, num_rays, layout, camera, ct, ci)
                        perf_ref = parse_stat("lbvh", scene, False, False, False, stat, ref_layout, camera, ref_ct, ref_ci)
                        perf = parse_stat(method, scene, splitting, annealing, sort, stat, layout, camera, ct, ci)
                        perfs.append(perf / perf_ref)
                        time = rays * 1.0e-6 / perf
                        time_ref = rays_ref * 1.0e-6 / perf_ref
                        times.append(time / time_ref)
                        ratio = time / cost
                        ratio_ref = time_ref / cost_ref
                        ratios.append(ratio / ratio_ref)
                avg_perf = np.array(perfs).mean()
                avg_time = np.array(times).mean()
                avg_ratio = np.array(ratios).mean()
                avg_tti = np.array(ttis).mean()
                g_perfs.append(avg_perf)
                g_times.append(avg_time)
                g_ratios.append(avg_ratio)
                g_ttis.append(avg_tti)
            g_times = np.array(g_times)
            g_ratios = np.array(g_ratios)
            time_mean = g_times.mean()
            best_val = np.max(g_perfs)
            if mode == 1:
                best_val = np.min(g_ttis)
            elif mode == 2:
                best_val = np.min(g_ratios)
            for method_index in range(len(methods)):
                if mode == 1:
                    value = g_ttis[method_index]
                    if value == best_val:
                        table.write(" & \\textbf{" + (precision2 % value) + "}")
                    else:
                        table.write(" & " + (precision2 % value))
                elif mode == 2:
                    value = g_ratios[method_index]
                    if value == best_val:
                        table.write(" & \\textbf{" + (precision2 % value) + "}")
                    else:
                        table.write(" & " + (precision2 % value))
                else:
                    value = g_perfs[method_index]
                    if value == best_val:
                        table.write(" & \\textbf{" + (precision2 % value) + "}")
                    else:
                        table.write(" & " + (precision2 % value))
            table.write("\\\\\n")


table.close()
