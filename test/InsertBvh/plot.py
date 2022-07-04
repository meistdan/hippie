import matplotlib.pyplot as plt
import os
import shutil
import seaborn as sns

scale = 0.375
width = scale * 16
height = scale * 9

sufix = "svg"

plot_dir = "./plot/"
directory = os.path.dirname(plot_dir)
shutil.rmtree(directory)
os.makedirs(directory)

benchmark_dir = "../../bin/benchmark/"

atrbvh = True

scenes = ["buddha", "sodahall", "hairball", "manuscript", "crown", "Pompeii", "san-miguel", "Vienna", "powerplant"]
#scene_indices = [0,1,2,3,4,5,6,7]
scene_indices = [5,6]
number_of_scenes = len(scene_indices)

mods = [1,4,9]
number_of_mods = len(mods)

atrs = [False,True]
number_of_atrs = len(atrs)

#strategies = [False,True]
strategies = [False]
number_of_strategies = len(strategies)

stat = "cost"
stat_label = "SAH cost [-]"

my_dpi = 96

sns.palplot(sns.color_palette("hls", 8))
sns.set_style("ticks")

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n + 1)

cmap = get_cmap(number_of_mods * number_of_atrs)
for s in range(number_of_scenes):
    si = scene_indices[s]
    scene = scenes[si]
    fig = plt.figure(figsize=(width, height))
    file_name = plot_dir + "/" + stat + "-" + scene + "-base." + sufix
    ax = fig.add_subplot(111)
    col = 0
    for atr in atrs:
        for m in range(number_of_mods):
            mod = mods[m]
            test_name = scene + "-insert-lbvh-aggr-" + str(mod) + "-0"
            method_label = "PRBVH$_{\\mu=" + str(mod) + "}^A$ (LBVH)"
            if atr:
                test_name = scene + "-insert-atr-aggr-" + str(mod) + "-0"
                method_label = "PRBVH$_{\\mu=" + str(mod) + "}^A$ (ATRBVH)"
            log_file = benchmark_dir + test_name + "/sts_" + test_name + ".log"
            print(log_file)
            with open(log_file) as file:
                data = file.read()
            data = data.split('\n')[:-1]
            x = [row.split(' ')[0] for row in data]
            y = [row.split(' ')[1] for row in data]
            ax.plot(x, y, label=method_label, marker='o')
            col = col + 1
    ax.legend()
    ax.grid(False)
    # ax.grid(linestyle=':', linewidth=0.5)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(stat_label)
    # plt.rc("axes.spines", top=False, right=False)
    plt.savefig(file_name, dpi=my_dpi)
    plt.close(fig)

cmap = get_cmap(number_of_mods * number_of_strategies)
for s in range(number_of_scenes):
    si = scene_indices[s]
    scene = scenes[si]
    fig = plt.figure(figsize=(width, height))
    file_name = plot_dir + "/" + stat + "-" + scene + "-str." + sufix
    ax = fig.add_subplot(111)
    col = 0
    for strategy in strategies:
        for m in range(number_of_mods):
            mod = mods[m]
            test_name = scene + "-insert-lbvh-aggr-" + str(mod) + "-0"
            method_label = "PRBVH$_{\\mu=" + str(mod) + "}^A$ (LBVH)"
            if strategy:
                test_name = scene + "-insert-lbvh-cons-" + str(mod) + "-0"
                method_label = "PRBVH$_{\\mu=" + str(mod) + "}^C$ (LBVH)"
            log_file = benchmark_dir + test_name + "/sts_" + test_name + ".log"
            print(log_file)
            with open(log_file) as file:
                data = file.read()
            data = data.split('\n')[:-1]
            x = [row.split(' ')[0] for row in data]
            y = [row.split(' ')[1] for row in data]
            ax.plot(x, y, label=method_label, marker='o')
            col = col + 1
    ax.legend()
    ax.grid(False)
    # ax.grid(linestyle=':', linewidth=0.5)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(stat_label)
    # plt.rc("axes.spines", top=False, right=False)
    plt.savefig(file_name, dpi=my_dpi)
    plt.close(fig)


