import numpy as np

FIN_PATH = "/user/HS229/mt00853/Documents/Codes/SFM_TEST/experiments/results/reconstruction/sfm_data.bin"
EXP_DIR = "/user/HS229/mt00853/Documents/Codes/SFM_TEST/experiments/"

def og_pipeline(input_dir = EXP_DIR + "test_images", output_dir = EXP_DIR + "results"):
    #!/usr/bin/python
    #! -*- encoding: utf-8 -*-
    # This file is part of OpenMVG (Open Multiple View Geometry) C++ library.
    # Python implementation of the bash script written by Romuald Perrot
    # Created by @vins31
    # Modified by Pierre Moulon
    #
    # this script is for easy use of OpenMVG
    #
    # usage : python openmvg.py image_dir output_dir
    #
    # image_dir is the input directory where images are located
    # output_dir is where the project must be saved
    #
    # if output_dir is not present script will create it
    #
    # Indicate the openMVG binary directory

    OPENMVG_SFM_BIN = "/user/HS229/mt00853/Documents/Codes/SFM_TEST/my_bin/bin"

    # Indicate the openMVG camera sensor width directory
    CAMERA_SENSOR_WIDTH_DIRECTORY = "/user/HS229/mt00853/Documents/Codes/SFM_TEST/openMVG/sfm_cpp/src/openMVG/exif/sensor_width_database"

    import os
    import subprocess

    matches_dir = os.path.join(output_dir, "matches")
    reconstruction_dir = os.path.join(output_dir, "original_pipeline_results")
    camera_file_params = os.path.join(CAMERA_SENSOR_WIDTH_DIRECTORY, "sensor_width_camera_database.txt")

    print ("Using input dir  : ", input_dir)
    print ("      output_dir : ", output_dir)

    # Create the ouput/matches folder if not present
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(matches_dir):
        os.mkdir(matches_dir)

    print ("1. Intrinsics analysis")
    pIntrisics = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_SfMInit_ImageListing"),  "-i", input_dir, "-o", matches_dir, "-d", camera_file_params] )
    pIntrisics.wait()

    print ("2. Compute features")
    pFeatures = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeFeatures"),  "-i", matches_dir+"/sfm_data.json", "-o", matches_dir, "-m", "SIFT"] )
    pFeatures.wait()

    print ("3. Compute matches")
    pMatches = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeMatches"),  "-i", matches_dir+"/sfm_data.json", "-o", matches_dir, "-g e"] )
    pMatches.wait()

    # Create the reconstruction if not present
    if not os.path.exists(reconstruction_dir):
        os.mkdir(reconstruction_dir)

    print ("4. Do Sequential/Incremental reconstruction")
    pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_GlobalSfM"),  "-i", matches_dir+"/sfm_data.json", "-m", matches_dir, "-o", reconstruction_dir] )
    pRecons.wait()

    print ("5. Colorize Structure")
    pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeSfM_DataColor"),  "-i", reconstruction_dir+"/sfm_data.bin", "-o", os.path.join(reconstruction_dir,"colorized.ply")] )
    pRecons.wait()

    # optional, compute final valid structure from the known camera poses
    print ("6. Structure from Known Poses (robust triangulation)")
    pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeStructureFromKnownPoses"),  "-i", reconstruction_dir+"/sfm_data.bin", "-m", matches_dir, "-f", os.path.join(matches_dir, "matches.f.bin"), "-o", os.path.join(reconstruction_dir,"robust.bin")] )
    pRecons.wait()

    pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeSfM_DataColor"),  "-i", reconstruction_dir+"/robust.bin", "-o", os.path.join(reconstruction_dir,"robust_colorized.ply")] )
    pRecons.wait()

def og_bundle():
    import sfm
    # Documents/Codes/
    i = EXP_DIR + "results/matches/sfm_data.json"
    m = EXP_DIR + "results/matches"
    o = EXP_DIR + "results/reconstruction"
    col = sfm.IntVector(500000)
    row = sfm.IntVector(500000)
    grad = sfm.DoubleVector(500000)
    o1 = sfm.DoubleVector()
    o2 = sfm.DoubleVector()
    rec = sfm.DoubleVector()

    sfm.fullBA(i, m, o, col, row, grad)

    print(col[100])
    print(np.array(col).shape)
    print(row[100])
    print(np.array(row).shape)
    print(grad[100])
    print(np.array(grad).shape)
    aaa = np.array(o1)
    bbb = np.array(o2)
    print(aaa[:12])
    print(aaa.shape)
    print(bbb[:12])
    print(bbb.shape)
    print(np.shape(rec))

if __name__ == "__main__":
    og_bundle()
