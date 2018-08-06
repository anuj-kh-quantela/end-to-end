import sys
import subprocess
import os
import shutil

if len(sys.argv) >= 2:

    # Generate det.npy
    if sys.argv[1] == "--generate":
        video_name = sys.argv[2]
        run_command_to_generate_npy = "python generate_detections.py --model=./resources/networks/mars-small128.ckpt-68577 --mot_dir=../data/" + video_name + "/ --output_dir=./resources/detections/" + video_name
        # print("Generating npy")
        # print("--------------------------------------------------")
        subprocess.call(run_command_to_generate_npy, shell=True)

    # Run racking
    elif sys.argv[1] == "--track":
        video_name = sys.argv[2]
        run_command_deep_sort_app = "python deep_sort_app.py --video_name=" + video_name
        # print("Start Tracking...")
        subprocess.call(run_command_deep_sort_app, shell=True)

        # dest_file_name = "./output/"+video_name+ "/"+"timeTagged-"+video_name+".avi"
        dest_file_name = "../intermediate_output/"+video_name+ "/"+"timeTagged-"+video_name+".avi"
        if os.path.isfile(dest_file_name):
            os.remove(dest_file_name)

        src_file_name = "timeTagged-"+video_name+".avi"
        if os.path.isfile(src_file_name):
            # shutil.move(src_file_name, "./output/"+video_name+ "/")
            shutil.move(src_file_name, "../intermediate_output/"+video_name+ "/")