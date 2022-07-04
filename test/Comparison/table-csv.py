import numpy as np

scene_path = "../../data/scenes"
benchmark_dir = "../../bin/benchmark/"

scenes = ["conference", "buddha", "sodahall", "hairball", "manuscript", "crown", "Pompeii", "san-miguel", "Vienna", "powerplant", "sponza", "crytek-sponza", "sibenik"]
scene_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
number_of_scenes = len(scene_indices)

sorts = [False, True]
layouts = ["bin", "quad", "oct"]
cameras = [0, 1, 2]

stat = "BVH COST"
# stat = "RT PERFORMANCE PATH"
precision = "%.2f"

table_file = "table-rt-cost.csv"
# table_file = "table-rt-perf.csv"
table = open(table_file, "w")
table.write(stat)


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


for s in range(number_of_scenes):
    si = scene_indices[s]
    scene = scenes[si]
    table.write(", " + scene)
table.write("\n")


def run(method, splitting, annealing, layout, sort):
    label = layout + "-" + method
    label += "-split" + str(int(splitting))
    label += "-anneal" + str(int(annealing))
    label += "-sort" + str(int(sort))
    table.write(label)
    for scene_index in scene_indices:
        scene = scenes[scene_index]
        value = 0
        for camera in cameras:
            test_name = scene + "-" + layout + "-" + method
            test_name += "-split" + str(int(splitting))
            test_name += "-anneal" + str(int(annealing))
            test_name += "-sort" + str(int(sort))
            test_name += "-cam" + str(int(camera))
            log_file = benchmark_dir + test_name + "/test_" + test_name + ".log"
            print(log_file)
            value += parse_stat(log_file, stat) / len(cameras)
        table.write(", " + (precision % value))
    table.write("\n")


for layout in layouts:
    for sort in sorts:
        run("lbvh", False, False, layout, sort)
        run("hlbvh", False, False, layout, sort)
        run("atr", False, False, layout, sort)
        run("ploc", False, False, layout, sort)
        run("sbvh", False, False, layout, sort)
        run("sbvh", True, False, layout, sort)
        run("insertion", False, False, layout, sort)
        run("insertion", False, True, layout, sort)
        run("insertion", True, False, layout, sort)
        run("insertion", True, True, layout, sort)

table.close()
