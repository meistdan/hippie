import numpy as np

benchmark_dir = "../../bin/benchmark/"

scenes = ["conference", "buddha", "sodahall", "hairball", "manuscript", "crown", "Pompeii", "san-miguel", "Vienna", "powerplant", "sponza", "crytek-sponza", "sibenik"]
scene_indices = [10, 12, 11, 0, 1, 2, 3, 5, 6, 7, 8, 9]
number_of_scenes = len(scene_indices)

sorts = [False, True]
layouts = ["bin", "quad", "oct"]
layout_labels = ["Binary", "Quaternary", "Octal"]
layout_labels = ["2", "4", "8"]
cameras = [0, 1, 2]
number_of_cameras = 3

stat_labels = ["Build time", "Trace time", "Time-to-image"]

methods = ["lbvh", "hlbvh", "ploc", "atr", "insertion", "insertion", "insertion", "insertion", "sbvh", "sbvh"]
annealings = [False, False, False, False, False, True, False, True, False, False]
splittings = [False, False, False, False, False, False, True, True, False, True]

precision0 = "%.0f"
precision2 = "%.2f"
ct = 3
ci = 2

table_file = "table-times-avg.tex"


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
    table.write("\\hline\n")
    table.write(layout_label)
    g_btimes = []
    g_ttimes = []
    g_ttis = []
    for method_index in range(len(methods)):
        method = methods[method_index]
        annealing = annealings[method_index]
        splitting = splittings[method_index]
        btimes = []
        ttimes = []
        ttis = []
        for scene_index in scene_indices:
            scene = scenes[scene_index]
            for camera in cameras:
                btime_label = "BVH CONSTRUCTION TIME " + ("KERNELS" if method_index <= 7 else "ABSOLUTE")
                btime_ref = parse_stat("lbvh", scene, False, False, False, btime_label, "bin", camera)
                btime = parse_stat(method, scene, splitting, annealing, False, btime_label, "bin", camera)
                btimes.append(btime / btime_ref)
                rays_ref = parse_stat("lbvh", scene, False, False, False, "NUMBER OF RAYS", ref_layout, camera)
                rays = parse_stat(method, scene, splitting, annealing, False, "NUMBER OF RAYS", layout, camera)
                perf_ref = parse_stat("lbvh", scene, False, False, False, "RT PERFORMANCE", ref_layout, camera)
                perf = parse_stat(method, scene, splitting, annealing, False, "RT PERFORMANCE", layout, camera)
                ttime = rays * 1.0e-6 / perf
                ttime_ref = rays_ref * 1.0e-6 / perf_ref
                ttimes.append(ttime / ttime_ref)
                tti = btime + ttime
                tti_ref = btime_ref + ttime_ref
                ttis.append(tti / tti_ref)
        avg_btime = np.array(btimes).mean()
        avg_ttime = np.array(ttimes).mean()
        avg_tti = np.array(ttis).mean()
        g_btimes.append(avg_btime)
        g_ttimes.append(avg_ttime)
        g_ttis.append(avg_tti)

    g_btimes = np.array(g_btimes)
    g_ttimes = np.array(g_ttimes)
    g_ttis = np.array(g_ttis)
    best_btime = np.min(g_btimes)
    best_ttime = np.min(g_ttimes)
    best_tti = np.min(g_ttis)

    table.write(" & Build time")
    for method_index in range(len(methods)):
        value = g_btimes[method_index]
        # if value == best_btime:
        #     table.write(" & \\textbf{" + (precision2 % value) + "}")
        # else:
        #     table.write(" & " + (precision2 % value))
        table.write(" & " + (precision2 % value))
    table.write("\\\\\n")

    table.write(" & Trace time")
    for method_index in range(len(methods)):
        value = g_ttimes[method_index]
        # if value == best_ttime:
        #     table.write(" & \\textbf{" + (precision2 % value) + "}")
        # else:
        #     table.write(" & " + (precision2 % value))
        table.write(" & " + (precision2 % value))
    table.write("\\\\\n")

    table.write(" & Time-to-image")
    for method_index in range(len(methods)):
        value = g_ttis[method_index]
        # if value == best_tti:
        #     table.write(" & \\textbf{" + (precision2 % value) + "}")
        # else:
        #     table.write(" & " + (precision2 % value))
        table.write(" & " + (precision2 % value))
    table.write("\\\\\n")

table.close()
