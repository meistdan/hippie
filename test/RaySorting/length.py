import numpy as np
import matplotlib.pyplot as plt
import os

scene_path = "../../data/scenes"
benchmark_dir = "../../bin/benchmark/"
output_dir = "./length/"

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
#ray_length = np.append(np.linspace(0.05, 0.4, 8), np.linspace(0.005, 0.05, 4))
ray_length = np.append(np.linspace(0.1, 0.4, 7),[0.0])
russian_roulette = [False]
morton_code_bits = [28, 32, 60]
morton_code_method = ["xxyyzz", "xyzxyz"]
hilbert_codes = [False, True]

dynamic_fetch = True

for hc in hilbert_codes:
    for rr in russian_roulette:
        for spp in samples_per_pixel:
            for r in recursion_depth:
                for s in scene_index:
                    scene = scenes[s]
                    for mb in morton_code_bits:
                        for mm in morton_code_method:
                            plt.figure()
                            speedups_overall = np.zeros(len(ray_length))
                            for c in range(number_of_cameras):
                                for li in range(len(ray_length)):
                                    l = ray_length[li]
                                    test_name = scene + "-cam=" + str(c) + "-rd=" + str(r) + "-spp=" + str(
                                        spp) + "-len=" + ('%.3f' % l) + "-mb=" + str(mb) + "-df=" + str(
                                        int(dynamic_fetch)) + "-hc=" + str(int(hc)) + "-rr=" + str(
                                        int(rr)) + "-mm=" + str(mm) + "-" + str(height) + "p"
                                    log_file = benchmark_dir + test_name + "/sort_" + test_name + ".log"
                                    print(log_file)
                                    sort_times = np.array([x.split(' ')[1] for x in open(log_file).readlines()]).astype(np.float32)
                                    trace_sort_times = np.array([x.split(' ')[2] for x in open(log_file).readlines()]).astype(np.float32)
                                    trace_time = np.array([x.split(' ')[3] for x in open(log_file).readlines()]).astype(np.float32)
                                    #speedups = (trace_time - (sort_times + trace_sort_times)) / trace_time
                                    speedups = (trace_time - trace_sort_times) / trace_time
                                    #speedups = np.clip(speedups, 0, 10)
                                    speedups_overall[li] += sum(speedups) / (r * number_of_cameras)
                            file_name = scene + "-rd=" + str(r) + "-spp=" + str(spp) + "-mb=" + str(mb) + "-rr=" + str(int(rr)) + "-hc=" + str(int(hc)) + "-mm=" + str(mm) + "-" + str(height) + "p"
                            plt.scatter(ray_length, speedups_overall)
                            plt.savefig(output_dir + "/" + file_name + ".png")
                            plt.close()
