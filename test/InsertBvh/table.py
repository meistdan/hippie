import numpy as np

scene_path = "../../data/scenes"
benchmark_dir = "../../bin/benchmark/"

scenes = ["buddha", "sodahall", "hairball", "manuscript", "crown", "Pompeii", "san-miguel", "Vienna", "powerplant"]
scene_indices = [0,1,2,3,4,5,6,7]
number_of_scenes = len(scene_indices)

stats = ["BVH COST", "RT PERFORMANCE", "RT PERFORMANCE PRIMARY", "RT PERFORMANCE SHADOW", "RT PERFORMANCE PATH", "BVH CONSTRUCTION TIME KERNELS"]
digits = [0,0,0,0,0,2]
number_of_stats = len(stats)

mods = [1,4,9]
number_of_mods = len(mods)
number_of_cameras = 3

atrs = [False]
number_of_atrs = len(atrs)

number_of_methods = number_of_atrs * number_of_mods + 4
values0 = np.zeros((number_of_methods, number_of_scenes))
values1 = np.copy(values0)
values2 = np.copy(values0)

def parse_stat(log_file, stat):
    file = open(log_file)
    for line in file:
        if stat in line:
            res = float(file.readline())
            file.close()
            return res
    file.close()
    print("stat was not found: " + stat)
    return 0

def value_to_str(v, vs):
    v_str = precision % v
    if t >= 1 and t <= 4:
        if round(np.max(vs[:, s]), digit) == round(v, digit):
            v_str = "\\textbf{" + str(v_str) + "}"
    else:
        if round(np.min(vs[:, s]), digit) == round(v, digit):
            v_str = "\\textbf{" + str(v_str) + "}"
    return v_str

def format_entry():
    v0 = values0[mi, s]
    v1 = values1[mi, s]
    v2 = values2[mi, s]
    if t == 4:
        return value_to_str(v1, values1) + " / " + value_to_str(v2, values2) + " / " + value_to_str(v0, values0)
    return value_to_str(v0, values0)

table_file = "table.tex"
table = open(table_file, "w")

for t in range(number_of_stats):

    stat = stats[t]
    digit = digits[t]
    precision = "%." + str(digit) + "f"

    for s in range(number_of_scenes):
        si = scene_indices[s]
        scene = scenes[si]
        value = 0
        for c in range(number_of_cameras):
            test_name = scene + "-lbvh-" + str(c)
            log_file = benchmark_dir + test_name + "/test_" + test_name + ".log"
            print(log_file)
            value += parse_stat(log_file, stat)
        value /= number_of_cameras
        mi = number_of_atrs * number_of_mods
        values0[mi, s] = value

    for s in range(number_of_scenes):
        si = scene_indices[s]
        scene = scenes[si]
        value = 0
        for c in range(number_of_cameras):
            test_name = scene + "-atr-" + str(c)
            log_file = benchmark_dir + test_name + "/test_" + test_name + ".log"
            print(log_file)
            value += parse_stat(log_file, stat)
        value /= number_of_cameras
        mi = number_of_atrs * number_of_mods + 1
        values0[mi, s] = value

    for s in range(number_of_scenes):
        si = scene_indices[s]
        scene = scenes[si]
        value = 0
        for c in range(number_of_cameras):
            test_name = scene + "-ploc-" + str(c)
            log_file = benchmark_dir + test_name + "/test_" + test_name + ".log"
            print(log_file)
            value += parse_stat(log_file, stat)
        value /= number_of_cameras
        mi = number_of_atrs * number_of_mods + 2
        values0[mi, s] = value

    for s in range(number_of_scenes):
        si = scene_indices[s]
        scene = scenes[si]
        value = 0
        if t < 5:
            for c in range(number_of_cameras):
                test_name = scene + "-rbvh-" + str(c)
                log_file = benchmark_dir + test_name + "/test_" + test_name + ".log"
                print(log_file)
                value += parse_stat(log_file, stat)
            value /= number_of_cameras
        else:
            log_file = scene_path + "/" + scene + "/" + scene + "-00c-spatial_median.log"
            value = parse_stat(log_file, "#BVH_OPTIMIZE_TIME")
        mi = number_of_atrs * number_of_mods + 3
        values0[mi, s] = value

    for a in range(number_of_atrs):
        atr = atrs[a]
        for m in range(number_of_mods):
            mod = mods[m]
            for s in range(number_of_scenes):
                si = scene_indices[s]
                scene = scenes[si]
                value = 0
                for c in range(number_of_cameras):
                    test_name = scene + "-insert-lbvh-aggr-" + str(mod) + "-" + str(c)
                    if atr:
                        test_name = scene + "-insert-atr-aggr-" + str(mod) + "-" + str(c)
                    log_file = benchmark_dir + test_name + "/test_" + test_name + ".log"
                    print(log_file)
                    value += parse_stat(log_file,stat)
                value /= number_of_cameras
                mi = m + a * number_of_mods
                values0[mi, s] = value

    if t == 2:
        values1 = np.copy(values0)
        continue

    if t == 3:
        values2 = np.copy(values0)
        continue

    method_label = "LBVH"
    table.write(method_label)
    for s in range(number_of_scenes):
        si = scene_indices[s]
        scene = scenes[si]
        mi = number_of_atrs * number_of_mods
        value = values0[mi, s]
        table.write(" & ")
        table.write(format_entry())
    table.write("\\\\\n")

    method_label = "ATRBVH"
    table.write(method_label)
    for s in range(number_of_scenes):
        si = scene_indices[s]
        scene = scenes[si]
        mi = number_of_atrs * number_of_mods + 1
        value = values0[mi, s]
        table.write(" & ")
        table.write(format_entry())
    table.write("\\\\\n")

    method_label = "PLOC"
    table.write(method_label)
    for s in range(number_of_scenes):
        si = scene_indices[s]
        scene = scenes[si]
        mi = number_of_atrs * number_of_mods + 2
        value = values0[mi, s]
        table.write(" & ")
        table.write(format_entry())
    table.write("\\\\\n")

    method_label = "RBVH"
    table.write(method_label)
    for s in range(number_of_scenes):
        si = scene_indices[s]
        scene = scenes[si]
        mi = number_of_atrs * number_of_mods + 3
        value = values0[mi, s]
        table.write(" & ")
        table.write(format_entry())
    table.write("\\\\\n")
    table.write("\\hline\n")

    for a in range(number_of_atrs):
        atr = atrs[a]
        for m in range(number_of_mods):
            mod = mods[m]
            method_label = "PRBVH$_{\\mu=" + str(mod) + "}^A$"
            if atr:
                method_label = "insert-atr-" + str(mod)
            table.write(method_label)
            for s in range(number_of_scenes):
                si = scene_indices[s]
                scene = scenes[si]
                mi = a * number_of_mods + m
                value = values0[mi, s]
                table.write(" & ")
                table.write(format_entry())
            table.write("\\\\\n")

    table.write("\n")

table.close()