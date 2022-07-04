import numpy as np
import matplotlib.pyplot as plt
import os

scene_path = "../../data/scenes"
benchmark_dir = "../../bin/benchmark/"
output_dir = "./dynamic_fetch/"

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
russian_roulette = [False]
morton_code_bits = [60]
morton_code_method = ["xxyyzz"]
hilbert_codes = [False]

dynamic_fetch = [False, True]

for spp in samples_per_pixel:
    for r in recursion_depth:
        for mb in morton_code_bits:
            for hc in hilbert_codes:
                for mm in morton_code_method:
                    for rr in russian_roulette:
                        for l in ray_length:
                            for s in scene_index:
                                scene = scenes[s]
                                trace_time_overall = np.zeros(len(dynamic_fetch) * 2)
                                labels = []
                                for di in range(len(dynamic_fetch)):
                                    df = dynamic_fetch[di]
                                    label = ("df=1" if df else "df=0") + "-sort=0"
                                    labels.append(label)
                                    label = ("df=1" if df else "df=0") + "-sort=1"
                                    labels.append(label)
                                    for c in range(number_of_cameras):
                                        test_name = scene + "-cam=" + str(c) + "-rd=" + str(r) + "-spp=" + str(spp) + "-len=" + ('%.3f' % l)\
                                            + "-mb=" + str(mb) + "-df=" + str(int(di)) + "-hc="  + \
                                                    str(int(hc)) + "-rr=" + str(int(rr)) + "-mm=" + str(mm) + "-" + str(height) + "p"
                                        log_file = benchmark_dir + test_name + "/sort_" + test_name + ".log"
                                        print(log_file)
                                        trace_sort_time = np.array([x.split(' ')[2] for x in open(log_file).readlines()]).astype(np.float32)
                                        trace_time = np.array([x.split(' ')[3] for x in open(log_file).readlines()]).astype(np.float32)
                                        trace_time_overall[di * 2 + 0] += sum(trace_time) / (r * number_of_cameras)
                                        trace_time_overall[di * 2 + 1] += sum(trace_sort_time) / (r * number_of_cameras)
                                file_name = scene + "-rd=" + str(r) + "-spp=" + str(spp) + "-len=" + (
                                    '%.3f' % l) + "-rr=" + str(int(rr)) + "-mb=" + str(mb) + "-mm=" + mm + "-hc=" + str(int(hc)) + "-" + str(height) + "p"
                                plt.figure()
                                plt.bar(range(len(dynamic_fetch) * 2), trace_time_overall)
                                plt.xticks(range(len(dynamic_fetch) * 2), labels)
                                plt.savefig(output_dir + "/" + file_name + ".png")
                                plt.close()