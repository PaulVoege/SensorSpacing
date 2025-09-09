# A Method for Evaluating the Effect of Infrastructure Sensor Spacing on Vehicle Detection Performance in Roadside Cooperative Sensing Networks

### Abstract
While detector models and onboard sensors of self-driving cars become increasingly capable, a
vehicle’s perspective remains inherently limited: large vehicles, buildings, foliage, and the like can
cause unpredictable occlusion-based gaps in perception. Where vehicle-based sensors are limited
to approximately ground level, infrastructure-based sensors can be placed higher above ground,
providing reliable coverage of potentially occluded areas. While clear that roadside cooperation
has potential to drastically improve vehicle detection performance, it is not currently clear how
densely roadside sensors should be placed to accomplish this efficiently.

This thesis investigates the effect of roadside sensor spacing on detection performance in a
cooperative detection network, making two primary contributions. First, this work presents a
method for generating, under identical traffic behaviour, multiple sets of cooperative roadside
Lidar and RGB data with incrementally increased roadside sensor spacing. Second, this work uses
the HEAL framework to evaluate the relationship between sensor spacing and detection
performance for 23 placements across 6 of CARLA’s Towns.

For each placement, the cooperative performance of 3 Lidar sensors across 70 one-meter
spacing increments is recorded at three thresholds for each of four pretrained detector models
provided by HEAL, resulting in 276 separate 70-point curves, each relating detection performance
to spacing.

An approximately proportional relationship between sensor spacing and detection
performance is identified for sensor spacings up to at least 70m. It is found that cooperative
roadside Lidar sensors can be expected to see an average improvement in detection performance
of 1.544 percent per meter of sensor spacing.

This repository contains the implementation of "A Method for Evaluating the Effect of Infrastructure Sensor Spacing
on Vehicle Detection Performance in Roadside Cooperative Sensing Networks"

### This repository contains:

-The traffic generation script `gen_traffic.py`

-The incrementally-spaced roadside data generation script `replay_with_sensors.py`

-The detection script's (`opencood/tools/inference.py`) associated automation script `autoinfer.py`

-Input and Output data for each placement

-The detector models used



## Environment
The following commands create a conda environment that handles the installation requirements that the requirements.txt won't handle.
```
conda create -n <your environment name> python=3.7
conda activate <your environment name>
pip3 install carla
pip3 install numpy
```
This will allow you to run python scripts as can be found in DataGenerator and Carla_0.9.15/PythonAPI/examples

To launch the CARLA simulator, run "Link to CarlaUE4-Linux-Shipping" in Carla_0.9.15 folder.

## Generate Traffic

In the DataGenerator folder:
```
python gen_traffic.py [-n <#vehicles>] [-w <#walkers>] [--seconds <runtime in seconds>]
```
For example
```
python gen_traffic.py -n 150 -w 70 --seconds 60
```
- defaults to 30 vehicles and 10 walkers unless otherwise specified
- Map is hardcoded in the .py file
- Capable of screen recording, if the option is enabled

## Generating Cooperative Roadside Data
In the DataGenerator folder:
```
python replay_with_sensors.py  -cfg <config ini file> [-skip <x>] -o "<output folder>"
```
For example, 
```
python replay_with_sensors.py -cfg /media/labuser/Data/DataGenerator/configparams/T1_1.ini -o "T1_1"
``` 
- -o denotes the output folder directory within       
- skip 44 can be used to skip, for example, 44 spacing instances. Useful when the simulator crashes midway through.
        
## Inference
The following command will run a single spacing instance through a specified detector model, and display the mAP performance for each threshold.

In the HEAL folder: 
```
python opencood/tools/inference.py --model_dir <directory of detector model> --fusion_method <intermediate> [--save_vis_interval <interval>] --test <dir>
``` 
For example:
```
python opencood/tools/inference.py --model_dir opencood/MyModel/HeterBaseline_opv2v_lidar_fcooper_2023_08_06_19_53_10/ --fusion_method intermediate --save_vis_interval 5 --test "output/T1_1"
```
### Autoinference
It was inefficient to manually use the inference script for each detector on all 1610 spacing instances, so the process was automated using the autoinfer.py script.
This runs from the DataGenerator folder, and consumes an entire placement, producing an inference measurement (mAP at 0.3, 0.5, and 0.7) for each spacing on each of the four detectors used.
In the DataGenerator folder:
```
python autoinfer.py [-fcooper] [-disco] [-attfuse] [-cobevt] -test "\<directory\>"
```
For example:
```
python autoinfer.py -fcooper -disco -attfuse -cobevt -test "output/T1_4"
```
        
### Gif-making
Within the detector model folder (HEAL/opencood/MyModel/\<detector\>), the detector
```
convert $(for ((a=0; a<995; a+=5)); do printf -- "-delay 30 bev_00%s.png " $a; done; ) result.gif

gifsicle --delay=15 --loop [5-8]00feb5.gif > anim.gif
```
