import numpy as np

scene_path = "../../data/scenes"
benchmark_dir = "../../bin/benchmark/"

scenes = ["sponza", "sibenik", "crytek-sponza", "conference", "buddha", "sodahall", "hairball", "manuscript", "crown", "Pompeii", "san-miguel", "Vienna", "powerplant"]
scene_indices = [0,1,2,3,4,6]
number_of_scenes = len(scene_indices)

stats = ["RT PERFORMANCE", "RT PERFORMANCE PRIMARY", "RT PERFORMANCE SHADOW", "RT PERFORMANCE PATH"]
digits = [0,0,0,0]
number_of_stats = len(stats)

number_of_cameras = 3

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

table_file = "table.tex"
table = open(table_file, "w")

for t in range(number_of_stats):

    stat = stats[t]
    digit = digits[t]
    precision = "%." + str(digit) + "f"

    method_label = "no-sort-2160p"
    table.write(method_label)
    for s in range(number_of_scenes):
        si = scene_indices[s]
        scene = scenes[si]
        value = 0
        for c in range(number_of_cameras):
            test_name = scene + "-" + str(c) + "-" + method_label
            log_file = benchmark_dir + test_name + "/test_" + test_name + ".log"
            print(log_file)
            value += parse_stat(log_file, stat)
        value /= number_of_cameras
        table.write(" & ")
        table.write(precision % value)
    table.write("\\\\\n")

    method_label = "no-sort-1s-2160p"
    table.write(method_label)
    for s in range(number_of_scenes):
        si = scene_indices[s]
        scene = scenes[si]
        value = 0
        for c in range(number_of_cameras):
            test_name = scene + "-" + str(c) + "-" + method_label
            log_file = benchmark_dir + test_name + "/test_" + test_name + ".log"
            print(log_file)
            value += parse_stat(log_file, stat)
        value /= number_of_cameras
        table.write(" & ")
        table.write(precision % value)
    table.write("\\\\\n")

    method_label = "sort-2160p"
    table.write(method_label)
    for s in range(number_of_scenes):
        si = scene_indices[s]
        scene = scenes[si]
        value = 0
        for c in range(number_of_cameras):
            test_name = scene + "-" + str(c) + "-" + method_label
            log_file = benchmark_dir + test_name + "/test_" + test_name + ".log"
            print(log_file)
            value += parse_stat(log_file, stat)
        value /= number_of_cameras
        table.write(" & ")
        table.write(precision % value)
    table.write("\\\\\n")

    method_label = "sort-1s-2160p"
    table.write(method_label)
    for s in range(number_of_scenes):
        si = scene_indices[s]
        scene = scenes[si]
        value = 0
        for c in range(number_of_cameras):
            test_name = scene + "-" + str(c) + "-" + method_label
            log_file = benchmark_dir + test_name + "/test_" + test_name + ".log"
            print(log_file)
            value += parse_stat(log_file, stat)
        value /= number_of_cameras
        table.write(" & ")
        table.write(precision % value)
    table.write("\\\\\n")

    method_label = "sort-a-2160p"
    table.write(method_label)
    for s in range(number_of_scenes):
        si = scene_indices[s]
        scene = scenes[si]
        value = 0
        for c in range(number_of_cameras):
            test_name = scene + "-" + str(c) + "-" + method_label
            log_file = benchmark_dir + test_name + "/test_" + test_name + ".log"
            print(log_file)
            value += parse_stat(log_file, stat)
        value /= number_of_cameras
        table.write(" & ")
        table.write(precision % value)
    table.write("\\\\\n")

    table.write("\n")

table.close()