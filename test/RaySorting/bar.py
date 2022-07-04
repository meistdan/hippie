import numpy as np
import matplotlib.pyplot as plt
import os

scene_path = "../../data/scenes"
benchmark_dir = "../../bin/benchmark/"
bar_dir = "./bar/"

width = 1920
height = 1080

if not os.path.exists(bar_dir):
    os.mkdir(bar_dir)

scenes = ["sponza", "sibenik", "crytek-sponza", "conference", "buddha", "sodahall", "hairball", "manuscript", "crown", "Pompeii", "san-miguel", "Vienna", "powerplant"]

number_of_cameras = 3

scene_index = [0, 1, 2, 3, 4, 6]
recursion_depth = [8]
samples_per_pixel = [32]
#ray_length = np.append(np.linspace(0.05, 0.4, 8), np.linspace(0.005, 0.05, 4))
ray_length = [0.005]
russian_roulette = [False, True]
morton_code_bits = [60]
morton_code_method = ["xxyyzz"]

for spp in samples_per_pixel:
    for r in recursion_depth:
        for s in scene_index:
            scene = scenes[s]
            for c in range(number_of_cameras):
                for mb in morton_code_bits:
                    for mm in morton_code_method:
                        for l in ray_length:
                            pi = 1
                            plt.figure(figsize=(16, 6))
                            for rr in russian_roulette:
                                test_name = scene + "-cam=" + str(c) + "-rd=" + str(r) + "-spp=" + str(spp) + "-len=" + (
                                '%.3f' % l) + "-mb=" + str(mb) + "-rr=" + str(int(rr)) + "-mm=" + str(mm) + "-" + str(height) + "p"
                                log_file = benchmark_dir + test_name + "/sort_" + test_name + ".log"
                                print(log_file)
                                avg_ray_counts = np.array([x.split(' ')[0] for x in open(log_file).readlines()]).astype(np.float32)
                                sort_times = np.array([x.split(' ')[1] for x in open(log_file).readlines()]).astype(np.float32)
                                trace_sort_times = np.array([x.split(' ')[2] for x in open(log_file).readlines()]).astype(np.float32)
                                trace_time = np.array([x.split(' ')[3] for x in open(log_file).readlines()]).astype(np.float32)
                                speedups_sort = (trace_time - (sort_times + trace_sort_times)) / trace_time
                                speedups = (trace_time - trace_sort_times) / trace_time
                                sort_performance =  avg_ray_counts / sort_times

                                plt.subplot(2, 4, pi)
                                plt.ylim(-0.3, 0.3)
                                plt.ylabel("speedup with sort")
                                plt.bar(range(r), speedups_sort)
                                pi += 1

                                plt.subplot(2, 4, pi)
                                plt.ylim(-0.3, 0.3)
                                plt.ylabel("speedup without sort")
                                plt.bar(range(r), speedups)
                                pi += 1

                                plt.subplot(2, 4, pi)
                                plt.ylabel("sort performance")
                                plt.bar(range(r), sort_performance)
                                pi += 1

                                plt.subplot(2, 4, pi)
                                plt.ylabel("avg. ray counts")
                                plt.bar(range(r), avg_ray_counts)
                                pi += 1

                            test_name = scene + "-cam=" + str(c) + "-rd=" + str(r) + "-spp=" + str(spp) + "-len=" + (
                                '%.3f' % l) + "-mb=" + str(mb) + "-mm=" + str(mm) + "-" + str(height) + "p"
                            plt.savefig(bar_dir + "/sort_" + test_name + ".png")
                            plt.close()
