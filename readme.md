This repository contains the implementation of "A Method for Evaluating the Effect of Infrastructure Sensor Spacing
on Vehicle Detection Performance in Roadside Cooperative Sensing Networks"

This repository contains:

-The traffic generation script

-The incrementally-spaced roadside data generation script

-The detection script, and associated 

-Input and Output data for each placement

-The detector models used




The following commands create a conda environment that handles the installation requirements that the requirements.txt won't handle.

conda create -n \<your environment name\> python=3.7

conda activate \<your environment name\>

pip3 install carla

pip3 install numpy

This will allow you to run python scripts as can be found in Myscripts and Carla_0.9.15/PythonAPI/examples

To launch the CARLA simulator, run "Link to CarlaUE4-Linux-Shipping" in Carla_0.9.15 folder.

## Generate Traffic

/media/labuser/Data/Myscripts$ "python gen_traffic.py [-n \<#vehicles\>] [-w \<#walkers\>] [--seconds \<runtime in seconds\>]"

    for example, $python gen_traffic.py -n 150 -w 70 --seconds 60
    
        -defaults to 30 vehicles and 10 walkers unless otherwise specified
        
        -Map is hardcoded in the .py file
        
        -Capable of screen recording, if the option is enabled

## Generating Cooperative Roadside Data

/media/labuser/Data/Myscripts$ python replay_with_sensors.py  -cfg \<config ini file\> [-skip <x>] -o "\<output folder\>"

   For example, $python replay_with_sensors.py -cfg /media/labuser/Data/Myscripts/configparams/T1_1.ini -o "T1_1" 
   
        -o is output folder name
        
        -skip 44 can be used to skip, for example, 44 spacing instances. Useful when the simulator crashes midway through.
        
## Inference
The following command will run a single spacing instance through a specified detector model, and display the mAP performance for each threshold.

In the HEAL folder: 
'''
python opencood/tools/inference.py --model_dir <directory of detector model> --fusion_method \<intermediate\> [--save_vis_interval \<interval\>] --test \<dir\>
''' 
For example:
'''
python opencood/tools/inference.py --model_dir opencood/MyModel/HeterBaseline_opv2v_lidar_fcooper_2023_08_06_19_53_10/ --fusion_method intermediate --save_vis_interval 5 --test "output/T1_1"
'''
### Autoinference

/media/labuser/Data/Myscripts$ python autoinfer.py [-fcooper] [-disco] [-attfuse] [-cobevt] -test "\<directory\>"

/media/labuser/Data/Myscripts$ python autoinfer.py -fcooper -disco -attfuse -cobevt -test "output/T1_4"
        
        
### Gif-making
Within the detector model folder (HEAL/opencood/MyModel/\<detector\>), the detector

convert $(for ((a=0; a<995; a+=5)); do printf -- "-delay 30 bev_00%s.png " $a; done; ) result.gif

gifsicle --delay=15 --loop [5-8]00feb5.gif > anim.gif
