#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# Also 
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Synchronized Lidar and Camera roadside data generator, using presaved traffic playback log files.
Recursively generates sensor data for the duration of playback, spacing sensors further apart each iteration.
Identical driving behaviour with different sensor spacing makes spacing-performance evaluation possible.
"""
ISWINDOWS=False

if ISWINDOWS:
    myscriptdir='E:\\'
else:
    myscriptdir='/media/labuser/Data/'

import glob
import os
import sys
import random
import csv
import math
import yaml
import time
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla # type: ignore
import open3d as o3d # type: ignore
import configparser
import argparse
from queue import Queue
from queue import Empty
from matplotlib import cm # type: ignore
from datetime import datetime

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    from PIL import Image
except ImportError:
    raise RuntimeError('cannot import PIL, make sure "Pillow" package is installed')

VIRIDIS = np.array(cm.get_cmap('viridis').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

class ParamFileReader:
    def __init__(self,fileNameInput,args):
        self.filename=fileNameInput
        config = configparser.ConfigParser()
        config.read(self.filename)

        #System Vars
        self.host=config.get("sysvars", "host")
        self.port=int(config.get("sysvars", "port"))

        #Simulation Vars
        self.mode=int(config.get('simvars','mode'))
        self.duration = int(config.get('simvars', "duration"))
        self.starttime = int(config.get("simvars", "starttime"))
        self.framemult = int(config.get("simvars", "framemult"))
        self.mapselect = config.get("simvars", "map")
        self.replayfile = config.get("simvars", "replayfile")
        self.anchorfile= config.get("simvars","anchorfile")
        self.quantity= int(config.get("simvars","iterations"))

        self.height=int(config.get("simvars","height"))
        # Camera Vars
        self.camx=int(config.get("camvars", "camx"))
        self.camy=int(config.get("camvars", "camy"))
        
        #Lidar Vars
        self.lidarfovupper=float(config.get("lidarvars", "upperfov"))
        self.lidarfovlower=float(config.get("lidarvars", "lowerfov"))
        self.lidarchannels=float(config.get("lidarvars", "channels"))
        self.lidarrange=float(config.get("lidarvars", "range"))
        self.lidarpointspersecond=int(config.get("lidarvars", "pointspersecond"))
        self.nonoise=config.getboolean("lidarvars", "nonoise")
        self.lidarrotationfreq=int(config.get("lidarvars","rotationfreq"))

        #Debug Flags
        self.lidardebug=config.getboolean("debugvars","lidardebug")
        self.framerecordmessage=config.getboolean("debugvars","frameRecordMessage")
        self.spawndebug=config.getboolean("debugvars","spawndebug")



        ##load sensor placement anchors from csv file specified by config file and store all the resultant sensor placements in a list
        print("[Info] Using Config File: "+self.filename)
        print("[Info] Using Anchor File: "+self.anchorfile)
        print("[Info] Using Replay File: "+self.replayfile)
        self.sensorsets=[]
        self.incsets=[]
        #anchorconfig = configparser.ConfigParser()
        #anchorconfig.read(self.anchorfile)
        
        #The following section reads sensor anchor locations and increments, but currently only one set
        with open(self.anchorfile) as f:
            anchordict = yaml.safe_load(f)
        
        #Base iteration settings
        baseloc=anchordict.get("base").get("location")
        baserot=anchordict.get("base").get("rotation")
        incloc=anchordict.get("increment").get("location")
        incrot=anchordict.get("increment").get("rotation")
        quant=anchordict.get("quant")

        #allow command line spacing override
        if args.xinc!=0:
            incloc[0]=args.xinc
        if args.yinc!=0:
            incloc[1]=args.yinc

        base=carla.Transform(carla.Location(float(baseloc[0]),float(baseloc[1]),float(baseloc[2])),carla.Rotation(float(baserot[0]),float(baserot[1]),float(baserot[2])))
        inc=carla.Transform(carla.Location(float(incloc[0]),float(incloc[1]),float(incloc[2])),carla.Rotation(float(incrot[0]),float(incrot[1]),float(incrot[2])))
        self.sensorsets.append([base,inc,quant])

        #Recursion Increment settings
        basechangeloc=anchordict.get("basechange").get("location")
        basechangerot=anchordict.get("basechange").get("rotation")
        additiveloc=anchordict.get("additive").get("location")
        additiverot=anchordict.get("additive").get("rotation")

        basechangeTF=carla.Transform(carla.Location(float(basechangeloc[0]),float(basechangeloc[1]),float(basechangeloc[2])),carla.Rotation(float(basechangerot[0]),float(basechangerot[1]),float(basechangerot[2])))
        additiveTF=carla.Transform(carla.Location(float(additiveloc[0]),float(additiveloc[1]),float(additiveloc[2])),carla.Rotation(float(additiverot[0]),float(additiverot[1]),float(additiverot[2])))
        self.incsets.append([basechangeTF,additiveTF,quant])

        self.base_spacing=math.hypot(incloc[0],incloc[1])#Uses hypoteneuse of ground plane coordinates to determine initial spacing.
        self.s_width=math.hypot(additiveloc[0],additiveloc[1]) #save incremental spacing also
                
    def getHost(self):
        return self.host
    def getPort(self):
        return self.port
    def getMode(self):
        return self.mode
    def getDuration(self):
        return self.duration
    def getStartTime(self):
        return self.starttime
    def getFrameMult(self):
        return self.framemult
    def getReplayFile(self):
        return self.replayfile
    def getSensorSets(self):
        return self.sensorsets
    def getIterations(self):
        return self.quantity
    def getincsets(self):
        return self.incsets
    def addToSensorSets(self,incsets):
        for j in range(len(incsets)):
            self.sensorsets[j][0]=addTF(self.sensorsets[j][0],incsets[j][0])
            self.sensorsets[j][1]=addTF(self.sensorsets[j][1],incsets[j][1])
    def getMapSelect(self):
        return self.mapselect
    def getCamX(self):
        return self.camx
    def getCamY(self):
        return self.camy
    def getLidarFovUpper(self):
        return self.lidarfovupper
    def getLidarFovLower(self):
        return self.lidarfovlower
    def getLidarChannels(self):
        return self.lidarchannels
    def getLidarRange(self):
        return self.lidarrange
    def getPointsPerSecond(self):
        return self.lidarpointspersecond
    def getNoNoise(self):
        return self.nonoise
    def getLidarRotationFreq(self):
        return self.lidarrotationfreq
    def getHeight(self):
        return self.height
    def getS_Width(self,i):
        sw=self.s_width*i
        sw=sw+self.base_spacing
        return sw
    def getLidarDebug(self):
        return self.lidardebug
    def getFrameRecordMessage(self):
        return self.framerecordmessage
    def getSpawnDebug(self):
        return self.spawndebug

    #def get(self):
    #    return self.


class SensorSuite:
    def __init__(self, id, egobp, cambp, lidbp):
        #variables
        self.id=id
        self.egobp=egobp
        self.cambp=cambp
        self.lidbp=lidbp
        self.ego=None
        self.camera=None
        self.lidar=None
        self.K=None
        self.kflag=False
        
    def cleanup(self):
        self.ego.destroy()
        self.camera.destroy()
        self.lidar.destroy()
        
    def spawn(self, world, spawntransform,freader):
        #spawn ego
        self.ego = world.spawn_actor(blueprint=self.egobp, transform=spawntransform)
        self.ego.set_collisions(False)
        self.ego.set_simulate_physics(False)
        h=freader.getHeight()
        #Camera
        self.camera = world.spawn_actor(blueprint=self.cambp, transform=carla.Transform(carla.Location(x=0, z=h)), attach_to=self.ego)#, carla.Rotation(pitch=-30)), attach_to=self.ego)
        #Lidar
        self.lidar = world.spawn_actor(blueprint=self.lidbp, transform=carla.Transform(carla.Location(x=0, z=h)), attach_to=self.ego)
        
        # The sensor data will be saved in thread-safe Queues
        self.image_queue = Queue()
        self.lidar_queue = Queue()
        self.camera.listen(lambda data: self.sensor_callback(data, self.image_queue))
        self.lidar.listen(lambda data: self.lidar_callback(data, self.lidar_queue))
        
    def getImageFrame(self):
            try:
                # Get the data once it's received.
                #might need to split the 'try' to ensure data frame synchronization, if cam misses an image, discard lidar?
                image_data = self.image_queue.get(True, 1.0)
                return image_data
            except Empty:
                print("[Warning] Some sensor data has been missed (image)")
                return
                
    def getLidarFrame(self):
            try:
                # Get the data once it's received.
                #might need to split the 'try' to ensure data frame synchronization, if cam misses an image, discard lidar?
                lidar_data = self.lidar_queue.get(True, 1.0)
                #print(lidar_data)
                return lidar_data
            except Empty:
                print("[Warning] Some sensor data has been missed (lidar)")
                return

    def getLidar(self):
        return self.lidar
    def getCamera(self):
        return self.camera
    def getEgo(self):
        return self.ego 
    def getK(self):
        ###IMAGE HANDLING STUFF
        if self.kflag==False:
            self.kflag=True
            # Build the K projection matrix:
            # K = [[Fx,  0, image_w/2],
            #      [ 0, Fy, image_h/2],
            #      [ 0,  0,         1]]
            image_w = self.cambp.get_attribute("image_size_x").as_int()
            image_h = self.cambp.get_attribute("image_size_y").as_int()
            fov = self.cambp.get_attribute("fov").as_float()
            focal = image_w / (2.0 * np.tan(fov * np.pi / 360.0))

            # In this case Fx and Fy are the same since the pixel aspect
            # ratio is 1
            K = np.identity(3)
            K[0, 0] = K[1, 1] = focal
            K[0, 2] = image_w / 2.0
            K[1, 2] = image_h / 2.0
            self.K=K
        ###IMAGE HANDLING STUFF
        return self.K
    def getimage_w(self):
        return self.cambp.get_attribute("image_size_x").as_int()
    def getimage_h(self):
        return self.cambp.get_attribute("image_size_y").as_int()

    def sensor_callback(self, data, queue):
        """This simple callback just stores the data on a thread safe Python Queue
        to be retrieved from the "main thread"."""
        queue.put(data)
        
    def lidar_callback(self, data, queue):
        """This simple callback just stores the data on a thread safe Python Queue
        to be retrieved from the "main thread"."""
        #if self.id==1:
        #    print(data)
        queue.put(data)
        
def addTF(tf1,tf2):
    loc=carla.Location(tf1.location.x + tf2.location.x,           tf1.location.y + tf2.location.y,          tf1.location.z + tf2.location.z)
    rot=carla.Rotation(pitch=tf1.rotation.pitch + tf2.rotation.pitch,   roll=tf1.rotation.roll + tf2.rotation.roll,    yaw=tf1.rotation.yaw + tf2.rotation.yaw)
    tfsum=carla.Transform(loc,rot)
    return tfsum

def generateAnchorSet(baseanchorTF,numAnchors,increment,anchormode):
    spawnpointList=[]
    #add base location
    spawnpointList.append(baseanchorTF)
    spawnpoint = baseanchorTF
    #DEBUG
    #print("[Debug] Anchor spawn: "+str(1)+"/"+str(numAnchors)+" (x:"+str(round(spawnpoint.location.x, 2))+", y:"+str(round(spawnpoint.location.y,2))+", z:"+str(round(spawnpoint.location.z,2))+")" )
    #xInc = 25
    #yInc = 0
    #zInc = 0
    if anchormode==1:#centered mode, 3 sensors centered on base anchor, with separations equal and opposite to the increment
        #Increment forwards
        spawnpoint=addTF(baseanchorTF,increment)
        spawnpointList.append(spawnpoint)
        #Invert increment transform
        invL=carla.Location(increment.location.x*-1,increment.location.y*-1,increment.location.z*-1)
        invR=carla.Rotation(pitch=increment.rotation.pitch*-1,roll=increment.rotation.roll*-1,yaw=increment.rotation.yaw*-1)
        #Increment backwards
        spawnpoint=addTF(baseanchorTF, carla.Transform(invL,invR))
        spawnpointList.append(spawnpoint)
    else:
        spawnpoint = baseanchorTF
        for i in range(numAnchors-1):#-1 to account for base point
            spawnpoint=addTF(spawnpoint,increment)
            spawnpointList.append(spawnpoint)
            #DEBUG
            #print("[Debug] Anchor spawn: "+str(i+2)+"/"+str(numAnchors)+" (x:"+str(round(spawnpoint.location.x,2))+", y:"+str(round(spawnpoint.location.y,2))+", z:"+str(round(spawnpoint.location.z,2))+")" )    
    return spawnpointList
    
def get_camera_intrinsic(sensor):
    """
    Retrieve the camera intrinsic matrix.

    Parameters
    ----------
    sensor : carla.sensor
        Carla rgb camera object.

    Returns
    -------
    matrix_x : np.ndarray
        The 2d intrinsic matrix.

    """
    VIEW_WIDTH = int(sensor.attributes['image_size_x'])
    VIEW_HEIGHT = int(sensor.attributes['image_size_y'])
    VIEW_FOV = int(float(sensor.attributes['fov']))

    matrix_k = np.identity(3)
    matrix_k[0, 2] = VIEW_WIDTH / 2.0
    matrix_k[1, 2] = VIEW_HEIGHT / 2.0
    matrix_k[0, 0] = matrix_k[1, 1] = VIEW_WIDTH / \
        (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))

    return matrix_k
    
def x_to_world_transformation(transform):
    """
    Get the transformation matrix from x(it can be vehicle or sensor)
    coordinates to world coordinate.

    Parameters
    ----------
    transform : carla.Transform
        The transform that contains location and rotation

    Returns
    -------
    matrix : np.ndarray
        The transformation matrx.

    """
    rotation = transform.rotation
    location = transform.location

    # used for rotation matrix
    c_y = np.cos(np.radians(rotation.yaw))
    s_y = np.sin(np.radians(rotation.yaw))
    c_r = np.cos(np.radians(rotation.roll))
    s_r = np.sin(np.radians(rotation.roll))
    c_p = np.cos(np.radians(rotation.pitch))
    s_p = np.sin(np.radians(rotation.pitch))

    matrix = np.identity(4)
    # translation matrix
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z

    # rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix

def matrix2list(matrix):
    """
    To generate readable yaml file, we need to convert the matrix
    to list format.

    Parameters
    ----------
    matrix : np.ndarray
        The extrinsic/intrinsic matrix.

    Returns
    -------
    matrix_list : list
        The matrix represents in list format.
    """

    assert len(matrix.shape) == 2
    return matrix.tolist()
    
def generateYamlDump(carlaworld,freader,suite):
    ##Setup 
    #collect vehicles
    data_dump=True
    world = carlaworld
    vehicle_list = world.get_actors().filter("*vehicle*")
    # todo: hard coded

    thresh=int(freader.getLidarRange())#ground truths are saved within lidar range
    #that said, it was hardcoded at 120 by default, and 50 for a lot of my testing

    #print('vehicle list size: %03d' %len(vehicle_list))
    #This line prunes the world vehicle list to those vehicles within <thresh> distance of the ego device
    #for each vehicle in the world, if the distance is less than thresh (50), and it's not just the ego device, keep it in the list
    v_list=[]
    #vehicle_list = [v for v in vehicle_list if v.get_location().distance(suite.getEgo().get_location()) < thresh]# and v.id != egoid]
    for v in vehicle_list:
        if v.get_location().distance(suite.getEgo().get_location()) <thresh:
            v_list.append(v)

    #Start saving
    #For each vehicle: (using the vehicle object type: carla.vehicle)
    #print('vehicle list size: %03d' %len(v_list))
    vehicle_dict={}
    dump_yml={}
    for vehicle in v_list:
    
        #preamble
        vel = vehicle.get_velocity()
        vel_meter_per_second = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        #above might need to be *3.6
        boundingbox=vehicle.bounding_box
        
        vehicle_dict.update({vehicle.id: {
            'bp_id': vehicle.type_id,
            'color': vehicle.attributes["color"] \
                if hasattr(vehicle, "attributes") \
                    and "color" in vehicle.attributes else None,
            #angle
            "angle": [vehicle.get_transform().rotation.roll,
                        vehicle.get_transform().rotation.yaw,
                        vehicle.get_transform().rotation.pitch],
            #center
            "center": [vehicle.bounding_box.location.x,
                        vehicle.bounding_box.location.y,
                        vehicle.bounding_box.location.z],
            #extent
            "extent": [vehicle.bounding_box.extent.x,
                        vehicle.bounding_box.extent.y,
                        vehicle.bounding_box.extent.z],
            #location
            "location": [vehicle.get_transform().location.x,
                            vehicle.get_transform().location.y,
                            vehicle.get_transform().location.z],
            #speed (vehicle.get_velocity() might work)
            "speed": vel_meter_per_second
        }})
    dump_yml.update({'vehicles': vehicle_dict})

    #true ego pose, maybe just use objective position of camera/lidar?
    #preamble
    ego_pos = suite.getEgo().get_location() #self.vehicle.get_transform()
    #ego_pos = suite.getLidar().get_location() #uses lidar position instead, as it's 4 units up in the air
    egospeed = 0#vehicle.get_velocity()    vel_meter_per_second = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
        #above might need to be *3.6
        #alternatively just clamp it to 0
    predicted_ego_pos = suite.getEgo().get_transform() #localization_manager.get_ego_pos()
    true_ego_pos = suite.getEgo().get_transform() #localization_manager.vehicle.get_transform()

    #true ego pose 
    dump_yml.update({'true_ego_pos': [
        true_ego_pos.location.x,
        true_ego_pos.location.y,
        true_ego_pos.location.z,
        true_ego_pos.rotation.roll,
        true_ego_pos.rotation.yaw,
        true_ego_pos.rotation.pitch]})
    #predicted ego pose
    dump_yml.update({'predicted_ego_pos': [
        predicted_ego_pos.location.x,
        predicted_ego_pos.location.y,
        predicted_ego_pos.location.z,
        predicted_ego_pos.rotation.roll,
        predicted_ego_pos.rotation.yaw,
        predicted_ego_pos.rotation.pitch]})
    #ego speed
    dump_yml.update({'ego_speed': float(egospeed)})

    #lidar pose
    lidartrans=suite.getLidar().get_transform()
    dump_yml.update({'lidar_pose': [
        lidartrans.location.x,
        lidartrans.location.y,
        lidartrans.location.z,
        lidartrans.rotation.roll,
        lidartrans.rotation.yaw,
        lidartrans.rotation.pitch]})
    #For each camera: #we only use 1 camera per installation
    camera_param = {}
    camera_trans = suite.getCamera().get_transform()
    camera_param.update({'cords': [
        camera_trans.location.x,
        camera_trans.location.y,
        camera_trans.location.z,
        camera_trans.rotation.roll,
        camera_trans.rotation.yaw,
        camera_trans.rotation.pitch
    ]})
    # dump intrinsic matrix
    camera_intrinsic = get_camera_intrinsic(suite.getCamera())
    camera_intrinsic = matrix2list(camera_intrinsic)
    camera_param.update({'intrinsic': camera_intrinsic})

    # dump extrinsic matrix lidar2camera
    lidar2world = x_to_world_transformation(lidartrans)
    camera2world = x_to_world_transformation(camera_trans)
    
    world2camera = np.linalg.inv(camera2world)
    lidar2camera = np.dot(world2camera, lidar2world)
    lidar2camera = matrix2list(lidar2camera)
    camera_param.update({'extrinsic': lidar2camera})
    dump_yml.update({'camera0': camera_param})

    dump_yml.update({'RSU': True})
    return dump_yml
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
###                                                             Main Function Entry                                                              ###
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
def Simulate(args):#filereader,iteration,outpathstatic):
        #load config file into storage object
    """
    if ISWINDOWS:
        cfile=myscriptdir+'Myscripts\configparams\config1.ini'#Windows
    else:
        cfile=myscriptdir+'Myscripts/configparams/config2linux.ini'#Linux
    """
    
    cfile=args.configfile
    filereader = ParamFileReader(cfile,args)
    if args.outpath=="0":
        outpathstatic=datetime.now()
        outpathstatic=outpathstatic.strftime("%Y-%m-%d_%H%M")
    else:
        outpathstatic=args.outpath
     # Connect to the server
    client = carla.Client(filereader.getHost(),filereader.getPort())
    client.set_timeout(30.0)#if your runs keep timing out, you may need to increase this:
    #I had to go from 5s->10s, and later 10s->30s. 
    #Loading things like the map can take a while, and may fool the script into thinking the simulator is absent, even on beefy machines
    #Notably an issue on Town10
    
    #Load Map specified by config: 'Town10'
    world = client.load_world(filereader.getMapSelect())
           
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

    original_settings = world.get_settings()
    outspacing=0
    #recursively run the simulations
    iterations=filereader.getIterations()
    print("[Info] Running "+str(iterations)+" simulation iterations.")
    allstarttime=time.time()
    for iteration in range(iterations):
        
        #export data_protocol.yaml for each instance
        #place it in each instance folder, outside of the individual sensor folders

        #values specified in anchorfile will be run first, 
            #then (iteration-1) increments of modified values. 
        #example: assuming x=10 with a x+=3 increment, 
            #it will result in x positions of 10, 13, 16, 19, 22 
            #during the first, second, third, fourth, and fifth run, respectively.
        
        if args.skip>iteration:#Skip the number of specified iterations
                filereader.addToSensorSets(filereader.getincsets())
                print("skipped "+str(iteration+1))
                continue
        if iteration>0: 
            filereader.addToSensorSets(filereader.getincsets())
            client.reload_world(reset_settings=False)

        s_width=filereader.getS_Width(iteration)
        print("\n[Start] ########################################")
        print("[Info] Running Simulation #"+str(iteration+1)+" of "+str(iterations)+"     Spacing: "+str(s_width)+"m.")
        outspacing=s_width

        start_time = time.time()
       
        world = client.get_world()#maybe redundant but it makes me feel better
        world.unload_map_layer(carla.MapLayer.ParkedVehicles)

        bp_lib = world.get_blueprint_library()

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        dataSuiteList=[]#setup empty array
        spawn_points=[]

        try:
            #make each run of the program have a separate output folder within the series, defined by
            if ISWINDOWS:
                outpath="output\\"+outpathstatic
                if not os.path.isdir(outpath):
                            os.mkdir(outpath)
                outpath=outpath+"\\"+str(iteration)
            else:
                outpath="output/"+outpathstatic
                if not os.path.isdir(outpath):
                            os.mkdir(outpath)
                outpath=outpath+"/"+str(iteration+1)
                if not os.path.isdir(outpath):
                    os.mkdir(outpath)
                #places each run in a subfolder in its own folder, to help with detection analysis
                outpath=outpath+"/1"
                if not os.path.isdir(outpath):
                    os.mkdir(outpath)
            ####################################################################################################################################################
            ###                                                             LOAD WORLD PLAYBACK                                                              ###
            ####################################################################################################################################################
            duration=filereader.getDuration() #duration is in seconds
            startTime=filereader.getStartTime() #usually takes ~10 frames for things to land on ground properly

            print("[Info] Traffic file: "+ filereader.getReplayFile())
            client.replay_file(filereader.getReplayFile(),startTime,duration,1)
            #client.replay_file("E:\Myscripts\Scenes\TrafficScene09_27_1451.log",startTime,duration,0)
            numFrames=int(duration/settings.fixed_delta_seconds)
            framemult=filereader.getFrameMult() #only record sensor data every 5th frame. 
                #at fixed_delta_seconds of 0.05, there are 20 frames per second. 
                #A framemult of 5 means that 4 frames (20/5=4) of sensor data will be recorded per second of the scenario.

            ####################################################################################################################################################
            ###                                                             SPAWN SENSOR SUITES                                                              ###
            ####################################################################################################################################################
            #list is the spawn locations
            #inputs: anchor, num, incrementTF
            #outputs: list of spawn point "carla.Transform"s
            sensorSets=filereader.getSensorSets()
            spectatortf=0
            for sensorset in sensorSets:
                baseanchorTF=sensorset[0]
                if spectatortf==0:
                    spectatortf=baseanchorTF
                incrementTF=sensorset[1]
                quantity=sensorset[2]

            
            anchormode=filereader.getMode()
            #Modes: 0='line' mode (any nondefined mode defaults to this setting)
            #       1='centered 3' mode
            #       2=N/A
            
            #baseanchorTF = carla.Transform(carla.Location(-32,5.2,0),carla.Rotation(0,90,0))
            #incrementTF = carla.Transform(carla.Location(25,0,0),carla.Rotation(0,0,0))

            for pnt in generateAnchorSet(baseanchorTF,quantity,incrementTF,anchormode):
                spawn_points.append(pnt)

            if not world.get_map().get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            
            ##CONFIGURATION
            # Search the desired blueprints
            vehicle_bp = bp_lib.filter("static.prop.trafficcone01")[0]
            camera_bp = bp_lib.filter("sensor.camera.rgb")[0]
            lidar_bp = bp_lib.filter("sensor.lidar.ray_cast")[0]
            
            # Configure the blueprints
            camera_bp.set_attribute("image_size_x", str(filereader.getCamX()))
            camera_bp.set_attribute("image_size_y", str(filereader.getCamY()))
            # Set up LIDAR , parameters are to assisst visualisation
            # THESE SETTINGS WORK
            lidar_bp = bp_lib.find('sensor.lidar.ray_cast') 
            lidar_bp.set_attribute('range', '100.0')
            lidar_bp.set_attribute('noise_stddev', '0.1')
            lidar_bp.set_attribute('upper_fov', '15.0')
            lidar_bp.set_attribute('lower_fov', '-25.0')
            lidar_bp.set_attribute('channels', '64.0')
            lidar_bp.set_attribute('rotation_frequency', '20.0')
            lidar_bp.set_attribute('points_per_second', '500000') 

            #THE FOLLOWING SETTINGS MAKE IT NOT WORK AT ALL HOLY HELL
            if filereader.getNoNoise():
                lidar_bp.set_attribute('dropoff_general_rate', '0.0')
                lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
                lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
            #lidar_bp.set_attribute('upper_fov', str(filereader.getLidarFovUpper()))
            #lidar_bp.set_attribute('lower_fov', str(filereader.getLidarFovLower()))
            #lidar_bp.set_attribute('channels', str(filereader.getLidarChannels()))
            #lidar_bp.set_attribute('range', str(filereader.getLidarRange))                     #<<<<<<THIS IS THE PROBLEM
            #lidar_bp.set_attribute('points_per_second', str(filereader.getPointsPerSecond))    #<<<<<<THIS IS THE PROBLEM
            #lidar_bp.set_attribute('rotation_frequency', str(filereader.getLidarRotationFreq))
            #sensor_fps=(1/settings.fixed_delta_seconds)#/framemult # sensor fps = global fps / framemult
            #lidar_bp.set_attribute("sensor_tick", str(settings.fixed_delta_seconds))
            ##END CONFIGURATION
            #for each defined spawn location, spawn in the whole sensor suite
            
            #DEBUG
            if filereader.getLidarDebug()==True:
                print("[Debug] Upper FOV:   "+str(lidar_bp.get_attribute('upper_fov').as_float())+"\tChannels: "+str(lidar_bp.get_attribute('channels').as_int())+"\t\tRotation Frequency: "+str(lidar_bp.get_attribute('rotation_frequency').as_float()))
                print("[Debug] Lower FOV:   "+str(lidar_bp.get_attribute('lower_fov').as_float())+"\tLidar Range: "+str(lidar_bp.get_attribute('range').as_float())+"\tLidar Points per Second: "+str(lidar_bp.get_attribute('points_per_second').as_int()))
            #print("[Debug] Nonoise: "+str(lidar_bp.get_attribute('range')))

            #DEBUG
            if filereader.getSpawnDebug()==True:
                sys.stdout.write('[Debug] Attempting sensor suite spawn: ')
            for systemid in range (len(spawn_points)):
                s=SensorSuite(systemid,vehicle_bp,camera_bp,lidar_bp)
                s.spawn(world, spawn_points[systemid], filereader)
                dataSuiteList.append(s)#for each spawn location, create a dataSuiteList object with appropriate blueprints
                #sys.stdout.write("%d/%d, " % (systemid+1,len(spawn_points)))    
                if filereader.getSpawnDebug()==True:
                    sys.stdout.write("x=%s:%d/%d, " % (spawn_points[systemid].location.x,systemid+1,len(spawn_points)))    
            sys.stdout.flush()
            ####################################################################################################################################################
            ###                                                        START FRAME-BY-FRAME ITERATION                                                        ###
            ####################################################################################################################################################
            
            #set spectator above and facing down towards base sensor
            spectator= world.get_spectator()
            spectatortf=carla.Transform(carla.Location(x=spectatortf.location.x,y=spectatortf.location.y,z=spectatortf.location.z+80),carla.Rotation(yaw=spectatortf.rotation.yaw,pitch=spectatortf.rotation.pitch-90,roll=spectatortf.rotation.roll))
            spectator.set_transform(spectatortf)

            #prime world and replayer with a single tick so actors are all loaded.
            world.tick()
            for suite in dataSuiteList:
                suite.getImageFrame()
                suite.getLidarFrame()
            #initial frame data from each sensor discarded as vehicles do not have collision during this tick, lidar will have produced bad data.

            #by default, replay sets all vehicles and pedestrians to have no collision.
            #this prevents lidars from interacting with them.
            #the following code manually activates collision and physics for all spawned actors
            all_actors=world.get_actors()
            for actor in all_actors:
                if "vehicle." in actor.type_id:
                    actor.set_collisions(True)
                    #actor.set_simulate_physics(True)
                if "walker." in actor.type_id:
                    actor.set_collisions(True)
                    #actor.set_simulate_physics(True)
            #now, when world.tick() prompts the lidars to generate detections, they will work.
            #print("[Info] Recording sensor data once every "+str(framemult)+" frame(s).")
            print("[Info] Starting playback for "+str(numFrames)+ " frames. " +str(int(numFrames/framemult))+" will be recorded in "+outpath+".")
            for frame in range(numFrames): #run the simulation for as many frames as is specified in argument
                world.tick()
                world_frame = world.get_snapshot().frame
                #print("[Debug] World Frame "+str(world_frame)+ ":")
                
                #pull data from sensor suites
                camlist=[]
                lidlist=[]
                ige=[]
                camige=[]
                lidige=[]
                detectedMismatch=False
                for suite in dataSuiteList: #for each instance of the sensor suite, extract the camera and lidar data frame.
                    cdat=suite.getImageFrame()
                    ldat=suite.getLidarFrame()
                    #Frame Mismatch Rectifier
                    #Ensures that all systems are operating on the same frame as the world frame.
                    #Potentially causes some minor desync issues between runs, with a maximum desync of framemult
                    #print("[Debug] Beforeframes: c_"+str(cdat.frame)+" l_"+str(ldat.frame)+" w_"+str(world_frame))
                    """
                    if (not(cdat.frame == ldat.frame == world_frame)):
                        if (not detectedMismatch):
                            sys.stdout.write("[Alert] Suites with frame mismatch:  %d" % (suite.id) )
                            #sys.stdout.flush()
                            detectedMismatch=True
                        else:
                            sys.stdout.write(",  %d" % (suite.id) )
                        while (ldat.frame<world_frame):
                            ldat=suite.getLidarFrame()
                        while (cdat.frame<world_frame):
                            cdat=suite.getImageFrame()
                    """
                    camlist.append(cdat)
                    lidlist.append(ldat)

                        #print("[Debug] Afterframes : c_"+str(cdat.frame)+" l_"+str(ldat.frame)+" w_"+str(world_frame))
                        #image_data->camlist[n]
                        #lidar_data->lidlist[n]
                    if detectedMismatch:
                        detectedMismatch=False
                        sys.stdout.write("\n")
                        sys.stdout.flush()

                    for k in range(len(camlist)):
                        if (not(camlist[k].frame == lidlist[k].frame == world_frame)):
                            print("[sad] frame desync: cam_"+str(camlist[k].frame)+" lid_"+str(lidlist[k].frame)+" world_"+str(world_frame))
                        assert camlist[k].frame == lidlist[k].frame == world_frame

                    # At this point, we have the synchronized frame information from the sensors across each instance of the suite.

                    
                if((frame%framemult)==0):
                    if filereader.getFrameRecordMessage()==True:
                        sys.stdout.write("\r(%d/%d) Recording Frame %d " % (1+(frame/framemult), int(numFrames/framemult), world_frame))# + ' \n')
                        sys.stdout.flush()
                    ############ Camera Handling #############
                    ##############    Start     ##############
                    
                    for k in range(len(camlist)):
                        # Get the raw BGRA buffer and convert it to an array of RGB of shape (image_data.height, image_data.width, 3).
                        im_array = np.copy(np.frombuffer(camlist[k].raw_data, dtype=np.dtype("uint8")))
                        im_array = np.reshape(im_array, (camlist[k].height, camlist[k].width, 4))
                        im_array = im_array[:, :, :3][:, :, ::-1]
                        #Save the ###camera### image using Pillow module into array of images.
                        camige.append(Image.fromarray(im_array))#sensor id is lost by here I think? but it might be the index of the entry anyways, so is it actually lost?
                    ##############     End      ##############
                    ############ Camera Handling #############


                    ##############    Start     ##############
                    ############## Save To File ##############
                    #08d means reserve 08 digits for the output and fill empty characters with zeros
                    #each sensor suite gets a subfolder of the output directory
                    #save mixed images
                    #setup output file structure
                    if not os.path.isdir(outpath):
                        os.mkdir(outpath)
                    #################################
                    #save ground truth bounding boxes
                    # to be replaced with a yaml dump in the style of opv2v and v2xset
                    """
                    if not os.path.isdir(outpath+'/gtruth'):
                        os.mkdir(outpath+'/gtruth')
                        #get bb, in the form of a list of 8x2 data points
                    bbs= ClientSideBoundingBoxes.get_bounding_boxes(world.get_actors().filter('vehicle.*'))
                    #save all bounding boxes for this frame
                    #csvrows=[]
                    csvpath=outpath+'/gtruth/'+str(frame)+'.csv'
                    with open(csvpath, mode='w', newline='') as file:
                        csvwriter=csv.writer(file,dialect='excel')  
                        for box in bbs:
                            r1=[]
                            for vertex in box:
                                r1.append(vertex.x)
                            r2=[]
                            for vertex in box:
                                r2.append(vertex.y)
                            r3=[]
                            for vertex in box:
                                r3.append(vertex.z)
                            r3.append(".")
                            csvwriter.writerow(r1)
                            csvwriter.writerow(r2)
                            csvwriter.writerow(r3)
                        file.close()

                    #save mixed cam/lidar images
                    #for k in range(len(ige)):
                    #    if not os.path.isdir(outpath+'/%03d' % k):
                    #        os.mkdir(outpath+'/%03d' % k) 
                    #    ige[k].save(outpath+"/%03d/m%06d.png" % (k, frame))
                    """
                    #############################
                    #save camera images
                    for k in range(len(camige)):
                        opath= outpath+('/%03d' % k)
                        #print(opath)
                        if not os.path.isdir(opath):
                            os.mkdir(opath) 
                        camige[k].save(opath+"/%06d_camera0.png" % (frame))

                    #############################
                    #save lidar pointclouds
                    for k in range(len(lidlist)):
                        opath= outpath+'/%03d' % k
                        if not os.path.isdir(opath):
                            os.mkdir(opath) 
                        #lidige[k].save(outpath+"/%03d/l%06d.png" % (k, frame))
                        #save lidar pointcloud
                        #print(lidlist[k].get_point_count(1))
                        lidlist[k].save_to_disk(opath+"/%06d.ply" % (frame))#needs to be changed to .pcd

                        pcloud=lidlist[k]
                        intense=[]
                        for xyzi in pcloud:
                            intense.append([xyzi.intensity,xyzi.intensity,xyzi.intensity])
                        #print(intense)

                        pcd = o3d.io.read_point_cloud(opath+"/%06d.ply" % (frame), format='ply')
                        #pcd=o3d.io.read_point_cloud_from_bytes(lidlist[k])

                        #intensity = np.zeros((np.size(pcd),3))
                        #intensity=[np.transpose(intense),np.transpose(intense),np.transpose(intense)]
                        #intensity[:,0] = np.reshape(intense,-1)
                        #intensity[:,1] = np.reshape(intense,-1)
                        #intensity[:,2] = np.reshape(intense,-1)
                        #print(intensity)
                        pcd.colors = o3d.utility.Vector3dVector(intense)
                        o3d.io.write_point_cloud(opath+"/%06d.pcd" % (frame), pcd,write_ascii=True)
   
                    #############################
                    #save yaml groundtruths
                    for k in range(len(dataSuiteList)):
                        #opath=outpath+'/%03d' % k
                        yamlout=generateYamlDump(world,filereader,dataSuiteList[k])
                        opath =outpath+ '/%03d/%06d.yaml' % (k,frame)
                        #save_path = os.path.join(opath,yml_name)
                        
                        if isinstance(yamlout, dict):
                            with open(opath, 'w') as outfile:
                                yaml.dump(yamlout, outfile, default_flow_style=False)
                    ##############     End      ##############
                    ############## Save To File ##############
            #END FRAME-BY-FRAME ITERATION
        #end try
        finally:     
            ltime=time.time() - start_time
            print("\r[End] ########################################")
            print("[End] Done "+str(int(outspacing))+"m in %.2f seconds! Estimated time to completion, %.2f minutes!" % (ltime,float(ltime*(iterations-(iteration+1))/60)))
            #print(str(outspacing)+"m.")
            

            # Destroy the actors in the scene.
            for suite in dataSuiteList:
                suite.cleanup()
            dataSuiteList=[]

            #client.apply_batch([carla.command.DestroyActor(x) for x in world.get_actors()])
    # End of Spacing Iterations
    world.apply_settings(original_settings)
    traffic_manager.set_synchronous_mode(False)
    
    # Apply the original settings when exiting.
    print("\n[End] ########################################")
    print("[End] Done all spacing instances in %.2f minutes!" % float((time.time()-allstarttime)/60))
    print("[End] ########################################")

def main():
    """Start function"""
    argparser = argparse.ArgumentParser(
        description='CARLA Sensor sync and projection tutorial')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='680x420',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '-f', '--frames',
        metavar='N',
        default=5,
        type=int,
        help='number of frames to record (default: 500)')
    argparser.add_argument(
        '-d', '--dot-extent',
        metavar='SIZE',
        default=4,
        type=int,
        help='visualization dot extent in pixels (Recomended [1-4]) (default: 2)')
    argparser.add_argument(
        '--no-noise',
        action='store_true',
        help='remove the drop off and noise from the normal (non-semantic) lidar')
    argparser.add_argument(
        '--upper-fov',
        metavar='F',
        default=30.0,
        type=float,
        help='lidar\'s upper field of view in degrees (default: 15.0)')
    argparser.add_argument(
        '--lower-fov',
        metavar='F',
        default=-25.0,
        type=float,
        help='lidar\'s lower field of view in degrees (default: -25.0)')
    argparser.add_argument(
        '-c', '--channels',
        metavar='C',
        default=64.0,
        type=float,
        help='lidar\'s channel count (default: 64)')
    argparser.add_argument(
        '-x', '--xinc',
        metavar='X',
        default=0,
        type=float,
        help='manual control of baseline x increment for sensor spacing')
    argparser.add_argument(
        '-y', '--yinc',
        metavar='Y',
        default=0,
        type=float,
        help='manual control of baseline y increment for sensor spacing')
    argparser.add_argument(
        '-r', '--range',
        metavar='R',
        default=100.0,
        type=float,
        help='lidar\'s maximum range in meters (default: 100.0)')
    argparser.add_argument(
        '-o', '--outpath',
        metavar='O',
        default="0",
        type=str,
        help='output folder name, defaults to current date/time')
    if ISWINDOWS:
        argparser.add_argument(
            '-cfg','--configfile',
            default='E:\Myscripts\configparams\config1.ini',#windows
            type=str)
    else:
        argparser.add_argument(
            '-cfg','--configfile',
            default='/media/labuser/Data/Myscripts/configparams/config2linux.ini',#linux
            type=str)
    argparser.add_argument(
        '--points-per-second',
        metavar='N',
        default='100000',
        type=int,
        help='lidar points per second (default: 100000)')
    argparser.add_argument('-skip','--skip',default=0,type=int)
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    args.dot_extent -= 1

    try:
        Simulate(args)
        #Automate()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
    
