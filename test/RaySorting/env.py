import numpy as np
import shutil
import os

env_dir = "./env/"
directory = os.path.dirname(env_dir)
shutil.rmtree(env_dir)
if (not os.path.exists(directory)):
    os.makedirs(directory)

width = 1920
height = 1080

scene_path = "../data/scenes"
scenes = ["sponza", "sibenik", "crytek-sponza", "conference", "buddha", "sodahall", "hairball", "manuscript", "crown",
          "Pompeii", "san-miguel", "Vienna", "powerplant"]

scene_index = [0, 1, 2, 3, 4, 6]
recursion_depth = [8]
samples_per_pixel = [8]
#ray_length = np.append(np.linspace(0.05, 0.4, 8), np.linspace(0.005, 0.05, 4))
#ray_length = np.linspace(0.2, 0.4, 5)
ray_length = np.linspace(0.1, 0.3, 5)
russian_roulette = [False]
morton_code_bits = [28, 32, 60]
#morton_code_bits = [60]
morton_code_method = ["xxyyzz", "xyzxyz"]
hilbert_codes = [False, True]

dynamic_fetch = True
images = True

sponza_light = "0.48 0.08 0.17"
sibenik_light = "0.48 0.08 0.17"
cryteksponza_light = "0.48 0.23 0.34"
conference_light = "0.48 0.08 0.17"
buddha_light = "0.0 0.45 0.0"
sodahall_light = "1.0 1.0 0.1"
hairball_light = "0.48 0.08 0.17"
manuscript_light = "0.4 0.31 0.39"
crown_light = "0.48 0.08 0.17"
pompeii_light = "0.87 0.96 0.92"
sanmiguel_light = "0.48 0.49 0.34"
vienna_light = "0.35 0.67 0.82"
powerplant_light = "0.0 0.59 0.0"

sponza_positions = ["0.91 0.11 0.23", "0.2 0.42 0.27", "0.51 0.08 0.39"]
sibenik_positions = ["0.15 0.08 0.2", "0.66 0.46 0.22", "0.33 0.03 0.26"]
cryteksponza_positions = ["0.2 0.08 0.31", "0.73 0.06 0.27", "0.3 0.36 0.31"]
conference_positions = ["0.18 0.08 0.14", "0.93 0.13 0.35", "0.3 0.12 0.59"]
buddha_positions = ["-0.15 0.64 -1.01", "0.06 0.72 -0.3", "-0.29 -0.11 -0.53"]
sodahall_positions = ["0.06 0.65 -0.23", "1.38 0.61 0.16", "0.74 0.96 -0.37"]
hairball_positions = ["1.1 -0.23 -0.48", "-0.47 0.42 -0.44", "-0.56 0.56 1.17"]
manuscript_positions = ["0.32 0.53 0.55", "0.33 0.7 1.25", "-0.25 0.8 -0.57"]
crown_positions = ["0.15 0.04 0.22", "0.34 0.07 0.22", "0.46 0.1 0.33"]
pompeii_positions = ["0.68 0.05 0.69", "0.53 0.03 0.55", "0.63 0.16 0.68"]
sanmiguel_positions = ["0.43 0.04 0.22", "0.51 0.2 0.33", "0.41 0.11 0.25"]
vienna_positions = ["0.8 0.01 0.31", "0.85 0.08 0.33", "0.49 0.51 0.6"]
powerplant_positions = ["0.01 0.15 0.27", "0.15 0.02 0.0", "0.14 0.23 0.05"]

sponza_directions = ["-1.0 0.03 -0.03", "0.91 -0.4 -0.1", "-0.01 0.05 -1.0"]
sibenik_directions = ["0.93 -0.06 0.37", "-0.91 -0.41 0.02", "0.88 0.4 -0.27"]
cryteksponza_directions = ["1.0 -0.04 0.02", "-0.94 0.33 0.12", "0.78 -0.63 -0.01"]
conference_directions = ["1.0 -0.02 0.02", "-0.99 -0.12 0.1", "0.01 -0.1 -1.0"]
buddha_directions = ["0.22 -0.11 0.97", "0.2 0.1 0.97", "0.44 0.54 0.71"]
sodahall_directions = ["0.49 -0.43 0.76", "-0.88 -0.34 0.34", "-0.21 -0.66 0.72"]
hairball_directions = ["-0.4 0.54 0.74", "0.74 0.07 0.67", "0.84 -0.04 -0.65"]
manuscript_directions = ["0.02 0.27 -0.96", "-0.02 -0.13 -0.99", "0.63 -0.21 0.74"]
crown_directions = ["0.76 0.2 0.62", "0.47 0.34 0.81", "0.18 0.14 0.97"]
pompeii_directions = ["-0.95 -0.31 -0.01", "-0.82 -0.36 -0.44", "-0.81 -0.58 -0.05"]
sanmiguel_directions = ["0.99 -0.13 -0.01", "0.13 -0.58 -0.8", "0.92 -0.19 -0.33"]
vienna_directions = ["-0.17 -0.1 0.98", "-0.87 -0.4 0.27", "-0.02 -0.97 -0.28"]
powerplant_directions = ["0.8 -0.34 -0.5", "0.82 0.32 0.47", "0.74 -0.64 0.22"]

number_of_cameras = len(conference_directions)

lights = [
    sponza_light,
    sibenik_light,
    cryteksponza_light,
    conference_light,
    buddha_light,
    sodahall_light,
    hairball_light,
    manuscript_light,
    crown_light,
    pompeii_light,
    sanmiguel_light,
    vienna_light,
    powerplant_light
]

positions = [
    sponza_positions,
    sibenik_positions,
    cryteksponza_positions,
    conference_positions,
    buddha_positions,
    sodahall_positions,
    hairball_positions,
    manuscript_positions,
    crown_positions,
    pompeii_positions,
    sanmiguel_positions,
    vienna_positions,
    powerplant_positions
]

directions = [
    sponza_directions,
    sibenik_directions,
    cryteksponza_directions,
    conference_directions,
    buddha_directions,
    sodahall_directions,
    hairball_directions,
    manuscript_directions,
    crown_directions,
    pompeii_directions,
    sanmiguel_directions,
    vienna_directions,
    powerplant_directions
]

for hc in hilbert_codes:
    for rr in russian_roulette:
        for spp in samples_per_pixel:
            for r in recursion_depth:
                for s in scene_index:
                    scene = scenes[s]
                    light = lights[s]
                    for c in range(number_of_cameras):
                        for mb in morton_code_bits:
                            for mm in morton_code_method:
                                for l in ray_length:
                                    test_name = scene + "-cam=" + str(c) + "-rd=" + str(r) + "-spp=" + str(spp) + "-len=" + ('%.3f' % l) + "-mb=" + str(mb) + "-df=" + str(int(dynamic_fetch)) + "-hc="  + str(int(hc)) + "-rr=" + str(int(rr)) + "-mm=" + str(mm) + "-" + str(height) + "p"
                                    print(test_name)
                                    env_file = env_dir + test_name + ".env"
                                    file = open(env_file, "w")
                                    file.write("Application {\n")
                                    file.write("mode benchmark\n")
                                    file.write("}\n")
                                    file.write("\n")
                                    file.write("Benchmark {\n")
                                    file.write("output " + test_name + "\n")
                                    file.write("test true\n")
                                    file.write("images " + str("true" if images else "false") + "\n")
                                    file.write("}\n")
                                    file.write("\n")
                                    file.write("Resolution {\n")
                                    file.write("width " + str(width) + "\n")
                                    file.write("height " + str(height) + "\n")
                                    file.write("}\n")
                                    file.write("\n")
                                    file.write("Scene {\n")
                                    file.write("filename " + scene_path + "/" + scene + "/" + scene + ".obj\n")
                                    file.write("light " + light + "\n")
                                    file.write("}\n")
                                    file.write("\n")
                                    file.write("Renderer {\n")
                                    file.write("rayType path\n")
                                    file.write("numberOfPrimarySamples " + str(spp) + "\n")
                                    file.write("recursionDepth " + str(r) + "\n")
                                    file.write("russianRoulette " + str("true" if rr else "false") + "\n")
                                    file.write("sortPathRays true\n")
                                    file.write("pathRayLength " + ('%.3f' % l) + "\n")
                                    file.write("pathMortonCodeMethod " + mm + "\n")
                                    file.write("pathMortonCodeBits " + str(mb) + "\n")
                                    file.write("pathHilbertCodes " + str("true" if hc else "false") + "\n")
                                    file.write("}\n")
                                    file.write("\n")
                                    file.write("Bvh {\n")
                                    file.write("adaptiveLeafSize true\n")
                                    file.write("method ploc\n")
                                    file.write("}\n")
                                    file.write("\n")
                                    file.write("Camera {\n")
                                    file.write("position " + positions[s][c] + "\n")
                                    file.write("direction " + directions[s][c] + "\n")
                                    file.write("fieldOfView 45.0\n")
                                    file.write("}\n")
                                    file.write("\n")
                                    file.close()
