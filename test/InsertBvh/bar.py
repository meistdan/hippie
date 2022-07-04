import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import seaborn as sns

scale = 0.375
width = scale * 16
height = scale * 9

sufix = "svg"
font_size = 8
rotation = 35

bar_width = 0.35
bar_offset = 1.5

phases = ["search", "lock nodes", "check locks", "reinsert", "refit", "compute cost"]

bar_dir = "./bar/"
directory = os.path.dirname(bar_dir)
shutil.rmtree(directory)
os.makedirs(directory)

benchmark_dir = "../../bin/benchmark/"

atrbvh = True

scenes = ["buddha", "sodahall", "hairball", "manuscript", "crown", "Pompeii", "san-miguel", "Vienna", "powerplant"]
#scene_indices = [0,1,2,3,4,5,6,7]
scene_indices = [6]
number_of_scenes = len(scene_indices)

#mods = [1,4,9]
mods = [1,4,9]
number_of_mods = len(mods)

atrs = [False,True]
number_of_atrs = len(atrs)

strategies = [False]
number_of_strategies = len(strategies)

stat = "time"
stat_label = "time [s]"

my_dpi = 96

sns.set_palette(sns.color_palette("Paired"))
# sns.set_palette(sns.color_palette("Blues"))
sns.set_style("ticks")

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n + 1)

method_labels = []
number_of_methods = number_of_mods * number_of_atrs
cmap = get_cmap(number_of_methods)
for s in range(number_of_scenes):
    si = scene_indices[s]
    scene = scenes[si]
    fig = plt.figure(figsize=(width, height))
    file_name = bar_dir + "/" + stat + "-" + scene + "-base." + sufix
    ax = fig.add_subplot(111)
    left = 0
    for atr in atrs:
        for m in range(number_of_mods):
            mod = mods[m]
            test_name = scene + "-insert-lbvh-aggr-" + str(mod) + "-0"
            method_label = "PRBVH$_{\\mu=" + str(mod) + "}^A$ (LBVH)"
            if atr:
                test_name = scene + "-insert-atr-aggr-" + str(mod) + "-0"
                method_label = "PRBVH$_{\\mu=" + str(mod) + "}^A$ (ATRBVH)"
            method_labels.append(method_label)
            log_file = benchmark_dir + test_name + "/phs_" + test_name + ".log"
            print(log_file)
            with open(log_file) as file:
                data = file.read()
            data = data.split('\n')[:-1]
            data = data[0].split(' ')
            data = np.array(data).astype('float')
            offset = 0
            bars = []
            for p in range(len(data)):
                bars.append(plt.barh(left, data[p], bar_width, bottom=offset))
                offset = offset + data[p]
            left = left + bar_offset
    ax.grid(False)
    ax.set_ylabel(stat_label)
    ax.legend(bars[::-1], phases[::-1])
    # plt.rc("axes.spines", top=False, right=False)
    plt.xticks(np.arange(0, number_of_methods * bar_offset, bar_offset), method_labels, fontsize=font_size, rotation=rotation)
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='on')
    plt.savefig(file_name, dpi=my_dpi)
    plt.close(fig)

method_labels = []
number_of_methods = number_of_mods * number_of_atrs
cmap = get_cmap(number_of_methods)
for s in range(number_of_scenes):
    si = scene_indices[s]
    scene = scenes[si]
    fig = plt.figure(figsize=(width, height))
    file_name = bar_dir + "/" + stat + "-" + scene + "-str." + sufix
    ax = fig.add_subplot(111)
    left = 0
    for strategy in strategies:
        for m in range(number_of_mods):
            mod = mods[m]
            test_name = scene + "-insert-lbvh-aggr-" + str(mod) + "-0"
            method_label = "PRBVH$_{\\mu=" + str(mod) + "}^A$ (LBVH)"
            if strategy:
                test_name = scene + "-insert-lbvh-cons-" + str(mod) + "-0"
                method_label = "PRBVH$_{\\mu=" + str(mod) + "}^C$ (LBVH)"
            method_labels.append(method_label)
            log_file = benchmark_dir + test_name + "/phs_" + test_name + ".log"
            print(log_file)
            with open(log_file) as file:
                data = file.read()
            data = data.split('\n')[:-1]
            data = data[0].split(' ')
            data = np.array(data).astype('float')
            offset = 0
            bars = []
            for p in range(len(data)):
                bars.append(plt.bar(left, data[p], bar_width, bottom=offset))
                offset = offset + data[p]
            left = left + bar_offset
    # ax.grid(linestyle=':', linewidth=0.5)
    ax.set_ylabel(stat_label)
    ax.legend(bars[::-1], phases[::-1])
    # plt.rc("axes.spines", top=False, right=False)
    plt.xticks(np.arange(0, number_of_methods * bar_offset, bar_offset), method_labels, fontsize=font_size, rotation=rotation)
    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='on')
    plt.savefig(file_name, dpi=my_dpi)
    plt.close(fig)