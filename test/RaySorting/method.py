import numpy as np
import matplotlib.pyplot as plt
import os

scene_path = "../../data/scenes"
benchmark_dir = "../../bin/benchmark/"
output_dir = "./method/"

width = 1920
height = 1080

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

scenes = ["sponza", "sibenik", "crytek-sponza", "conference", "buddha", "sodahall", "hairball", "manuscript", "crown", "Pompeii", "san-miguel", "Vienna", "powerplant"]
number_of_scenes = len(scenes)

number_of_cameras = 3

scene_index = [0, 1, 2, 3, 4, 6]
recursion_depth = [8]
samples_per_pixel = [32]
ray_length = [0.25]
russian_roulette = [False, True]
morton_code_bits = [28]
morton_code_method = ["xyzxyz", "xxyyzz"]

for spp in samples_per_pixel:
    for r in recursion_depth:
        for mb in morton_code_bits:
            for rr in russian_roulette:
                for l in ray_length:
                    for s in scene_index:
                        scene = scenes[s]
                        speedups_overall = np.zeros(len(morton_code_method))
                        for mi in range(len(morton_code_method)):
                            mm = morton_code_method[mi]
                            for c in range(number_of_cameras):
                                test_name = scene + "-cam=" + str(c) + "-rd=" + str(r) + "-spp=" + str(spp) + "-len=" + (
                                '%.3f' % l) + "-mb=" + str(mb) + "-rr=" + str(int(rr)) + "-mm=" + str(mm) + "-" + str(height) + "p"
                                log_file = benchmark_dir + test_name + "/sort_" + test_name + ".log"
                                print(log_file)
                                sort_times = np.array([x.split(' ')[1] for x in open(log_file).readlines()]).astype(np.float32)
                                trace_sort_times = np.array([x.split(' ')[2] for x in open(log_file).readlines()]).astype(np.float32)
                                trace_time = np.array([x.split(' ')[3] for x in open(log_file).readlines()]).astype(np.float32)
                                speedups = (trace_time - (sort_times + trace_sort_times)) / trace_time
                                speedups = np.clip(speedups, 0, 10)
                                speedups_overall[mi] += sum(speedups) / (r * number_of_cameras)
                        file_name = scene + "-rd=" + str(r) + "-spp=" + str(spp) + "-len=" + (
                            '%.3f' % l) + "-mb=" + str(mb) + "-rr=" + str(int(rr)) + "-" + str(height) + "p"
                        plt.figure()
                        plt.bar(range(len(morton_code_method)), speedups_overall)
                        plt.xticks(range(len(morton_code_method)), morton_code_method)
                        plt.savefig(output_dir + "/" + file_name + ".png")
                        plt.close()