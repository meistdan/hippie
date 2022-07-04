import os
import shutil

env_dir = "./env/"
directory = os.path.dirname(env_dir)
if os.path.isdir(env_dir):
    shutil.rmtree(directory)
os.makedirs(directory)

scene_path = "../data/scenes"
scenes = ["conference", "buddha", "sodahall", "hairball", "manuscript", "crown", "Pompeii", "san-miguel", "Vienna", "powerplant", "sponza", "crytek-sponza", "sibenik", "gallery"]
# scene_indices = [0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13]
scene_indices = [13]
iterations = 3000

recursion_depth = 8
# samples = 32
samples = 1024
images = True

sorts = [False, True]
layouts = ["bin", "quad", "oct"]
cameras = [0, 1, 2]
cs = [[3, 2]]

conference_light="0.48 0.08 0.17"
buddha_light="0.0 0.45 0.0"
sodahall_light="1.0 1.0 0.1"
hairball_light="0.48 0.08 0.17"
manuscript_light="0.4 0.31 0.39"
crown_light="0.48 0.08 0.17"
pompeii_light="0.87 0.96 0.92"
sanmiguel_light="0.48 0.49 0.34"
vienna_light="0.35 0.67 0.82"
powerplant_light="0.0 0.59 0.0"
sponza_light="0.54 0.25 0.22"
cryteksponza_light="0.29 0.16 0.33"
sibenik_light="0.60 0.43 0.23"
gallery_light="0.14 0.09 0.88"

conference_positions = ["0.18 0.08 0.14", "0.93 0.13 0.35", "0.3 0.12 0.59"]
buddha_positions = ["-0.15 0.64 -1.01", "0.06 0.72 -0.3", "-0.29 -0.11 -0.53"]
sodahall_positions = ["0.06 0.65 -0.23", "1.38 0.61 0.16", "0.74 0.96 -0.37"]
hairball_positions = ["1.1 -0.23 -0.48", "-0.47 0.42 -0.44", "0.10 0.54 0.85"]
manuscript_positions = ["0.32 0.53 0.55", "0.33 0.7 1.25", "-0.25 0.8 -0.57"]
crown_positions = ["0.15 0.04 0.22", "0.34 0.07 0.22", "0.46 0.1 0.33"]
pompeii_positions = ["0.68 0.05 0.69", "0.53 0.03 0.55", "0.63 0.16 0.68"]
sanmiguel_positions = ["0.43 0.04 0.22", "0.51 0.2 0.33", "0.41 0.11 0.25"]
vienna_positions = ["0.8 0.01 0.31", "0.85 0.08 0.33", "0.49 0.51 0.6"]
powerplant_positions = ["0.01 0.15 0.27", "0.15 0.02 0.0", "0.14 0.23 0.05"]
sponza_positions = ["0.18 0.09 0.24", "0.47 0.39 0.23", "0.50 0.25 0.38"]
cryteksponza_positions = ["0.26 0.06 0.32", "0.26 0.06 0.32", "0.76 0.07 0.31"]
sibenik_positions = ["0.18 0.08 0.14", "0.23 0.23 0.22", "0.74 0.32 0.20"]
gallery_positions = ["0.15 0.11 0.77", "0.37 0.16 0.23", "0.14 0.09 0.88"]

conference_directions = ["1.0 -0.02 0.02", "-0.99 -0.12 0.1", "0.01 -0.1 -1.0"]
buddha_directions = ["0.22 -0.11 0.97", "0.2 0.1 0.97", "0.44 0.54 0.71"]
sodahall_directions = ["0.49 -0.43 0.76", "-0.88 -0.34 0.34", "-0.21 -0.66 0.72"]
hairball_directions = ["-0.4 0.54 0.74", "0.74 0.07 0.67", "0.74 -0.12 -0.66"]
manuscript_directions = ["0.02 0.27 -0.96", "-0.02 -0.13 -0.99", "0.63 -0.21 0.74"]
crown_directions = ["0.76 0.2 0.62", "0.47 0.34 0.81", "0.18 0.14 0.97"]
pompeii_directions = ["-0.95 -0.31 -0.01", "-0.82 -0.36 -0.44", "-0.81 -0.58 -0.05"]
sanmiguel_directions = ["0.99 -0.13 -0.01", "0.13 -0.58 -0.8", "0.92 -0.19 -0.33"]
vienna_directions = ["-0.17 -0.1 0.98", "-0.87 -0.4 0.27", "-0.02 -0.97 -0.28"]
powerplant_directions = ["0.8 -0.34 -0.5", "0.82 0.32 0.47", "0.74 -0.64 0.22"]
sponza_directions = ["1.0 0.0 0.0", "-0.81 -0.59 -0.02", "0.99 -0.10 -0.02"]
cryteksponza_directions = ["1.0 0.0 0.0", "0.82 0.32 0.47", "1.00 0.07 0.01"]
sibenik_directions = ["1.0 -0.02 0.02", "0.93 -0.37 -0.01", "-0.87 -0.49 0.01"]
gallery_directions = ["0.0 -0.01 1.0", "-0.38 -0.18 0.91", "0.36 -0.05 -0.93"]

number_of_cameras = len(conference_directions)

lights = [
conference_light,
buddha_light,
sodahall_light,
hairball_light,
manuscript_light,
crown_light,
pompeii_light,
sanmiguel_light,
vienna_light,
powerplant_light,
sponza_light,
cryteksponza_light,
sibenik_light,
gallery_light
]

positions = [
conference_positions,
buddha_positions,
sodahall_positions,
hairball_positions,
manuscript_positions,
crown_positions,
pompeii_positions,
sanmiguel_positions,
vienna_positions,
powerplant_positions,
sponza_positions,
cryteksponza_positions,
sibenik_positions,
gallery_positions
]

directions = [
conference_directions,
buddha_directions,
sodahall_directions,
hairball_directions,
manuscript_directions,
crown_directions,
pompeii_directions,
sanmiguel_directions,
vienna_directions,
powerplant_directions,
sponza_directions,
cryteksponza_directions,
sibenik_directions,
gallery_directions
]


def output_settings(scene_index, method, splitting, annealing, layout, sort, camera, ct, ci):
    scene = scenes[scene_index]
    light = lights[scene_index]
    test_name = scene + "-" + layout + "-" + method
    test_name += "-split" + str(int(splitting))
    test_name += "-anneal" + str(int(annealing))
    test_name += "-sort" + str(int(sort))
    test_name += "-cam" + str(int(camera))
    test_name += "-ct" + str(ct)
    test_name += "-ci" + str(ci)
    # test_name += "-sl"
    # test_name = scene
    # test_name += str(int(camera))
    print(test_name)
    env_file = env_dir + test_name + ".env"
    file = open(env_file, "w")
    file.write("Application {\n")
    file.write("mode benchmark\n")
    file.write("}\n")
    file.write("\n")
    file.write("Resolution {\n")
    file.write("width 512\n")
    file.write("height 384\n")
    file.write("}\n")
    file.write("\n")
    file.write("Benchmark {\n")
    file.write("output " + test_name + "\n")
    if images:
        file.write("images true\n")
    file.write("}\n")
    file.write("\n")
    file.write("Scene {\n")
    file.write("filename " + scene_path + "/" + scene + "/" + scene + ".obj\n" )
    file.write("light " + light + "\n")
    file.write("}\n")
    file.write("\n")
    file.write("Renderer {\n")
    file.write("rayType path\n")
    file.write("numberOfPrimarySamples " + str(samples) + "\n")
    file.write("recursionDepth " + str(recursion_depth) + "\n")
    file.write("sortPathRays " + ("true" if sort else "false") + "\n")
    file.write("sortShadowRays " + ("true" if sort else "false") + "\n")
    file.write("whitePoint 1.8\n")
    file.write("keyValue 1.0\n")
    file.write("}\n")
    file.write("\n")
    file.write("Bvh {\n")
    file.write("ct " + str(ct) + "\n")
    file.write("ci " + str(ci) + "\n")
    file.write("layout " + layout + "\n")
    file.write("adaptiveLeafSize true\n")
    file.write("method " + method + "\n")
    file.write("presplitting false\n")
    if not splitting:
        file.write("sbvhAlpha 1.5\n")
    file.write("insertionSbvh " + ("true" if splitting else "false") + "\n")
    file.write("insertionAnnealing " + ("true" if annealing else "false") + "\n")
    file.write("insertionAnnealingIterations " + str(iterations) + "\n")
    file.write("plocRadius 100\n")
    file.write("atrIterations 20\n")
    file.write("}\n")
    file.write("\n")
    file.write("Camera {\n")
    file.write("position " + positions[scene_index][camera] + "\n")
    file.write("direction " + directions[scene_index][camera] + "\n")
    file.write("fieldOfView 45.0\n")
    file.write("}\n")
    file.write("\n")
    file.close()


# for sort in sorts:
#     for layout in layouts:
#         for scene_index in scene_indices:
#             for camera in cameras:
#                 for c in cs:
#                     ct = c[0]
#                     ci = c[1]
                    # output_settings(scene_index, "lbvh", False, False, layout, sort, camera, ct, ci)
                    # output_settings(scene_index, "hlbvh", False, False, layout, sort, camera, ct, ci)
                    # output_settings(scene_index, "atr", False, False, layout, sort, camera, ct, ci)
                    # output_settings(scene_index, "ploc", False, False, layout, sort, camera, ct, ci)
                    # output_settings(scene_index, "sbvh", False, False, layout, sort, camera, ct, ci)
                    # output_settings(scene_index, "sbvh", True, False, layout, sort, camera, ct, ci)
                    # output_settings(scene_index, "insertion", False, False, layout, sort, camera, ct, ci)
                    # output_settings(scene_index, "insertion", False, True, layout, sort, camera, ct, ci)
                    # output_settings(scene_index, "insertion", True, False, layout, sort, camera, ct, ci)
                    # output_settings(scene_index, "insertion", True, True, layout, sort, camera, ct, ci)

# output_settings(3, "insertion", False, False, "oct", True, 1)

for scene_index in scene_indices:
    for camera in cameras:
        for c in cs:
            ct = c[0]
            ci = c[1]
            output_settings(scene_index, "lbvh", False, False, "bin", False, camera, ct, ci)