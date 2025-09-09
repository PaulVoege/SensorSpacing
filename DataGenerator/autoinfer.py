#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""

"""


import glob
import os
import time

import sys
#This directory may need to be reconfigured depending on local directory structure:
#ensure this directory is targetted properly
sys.path.append('/media/labuser/Data/SensorSpacing/HEAL/opencood/tools')

import argparse
import inferenceCustom as detector 


def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    #argparser.add_argument('--model', type=str,default='')
    argparser.add_argument('--test','-test', type=str,default='')
    argparser.add_argument('--do_fcooper','-fcooper',action='store_true',default=False)
    argparser.add_argument('--do_cobevt','-cobevt',action='store_true',default=False)
    argparser.add_argument('--do_attfuse','-attfuse',action='store_true',default=False)
    argparser.add_argument('--do_disco','-disco',action='store_true',default=False)
    argparser.add_argument('--saveinterval',type=int,default=10)
    argparser.add_argument('--number','-n',type=int,default=70)
    args = argparser.parse_args()

    #python autoinfer.py -do_fcooper -n 4 -test "output/T3_3" 

    #ensure these directories are targetted properly
    fcoopermodeldir="/media/labuser/Data/HEAL/opencood/MyModel/HeterBaseline_opv2v_lidar_fcooper_2023_08_06_19_53_10/"
    cobevtmodeldir="/media/labuser/Data/HEAL/opencood/MyModel/HeterBaseline_opv2v_lidar_cobevt_2023_08_20_18_06_40"
    attfusemodeldir="/media/labuser/Data/HEAL/opencood/MyModel/HeterBaseline_opv2v_lidar_attfuse_2023_08_06_19_58_00"
    discomodeldir="/media/labuser/Data/HEAL/opencood/MyModel/HeterBaseline_opv2v_lidar_disco_2023_08_06_20_01_40"

    fcoopercsvfile=args.test+"/fcooper.csv"
    cobevtcsvfile=args.test+"/cobevt.csv"
    attfusecsvfile=args.test+"/attfuse.csv"
    discocsvfile=args.test+"/disco.csv"
    TotalTimeStart=time.time()
    for i in range(args.number):
        looptime=time.time()
        test=args.test+"/"+str(i+1)
        print("["+str(i+1)+"/"+str(args.number)+"] Running detections on "+test)
        if args.do_fcooper:
            print("###########################################################################")
            print("["+str(i+1)+"/"+str(args.number)+"] Running Detections on fcooper: ")
            print("###########################################################################")
            start_time=time.time()
            detector.infer(fcoopermodeldir,fcoopercsvfile,test,args.saveinterval)
            print("###########################################################################")
            fcoopertime=(time.time() - start_time)
            print("["+str(i+1)+"/"+str(args.number)+"] Done fcooper in %.2f seconds!" % fcoopertime)
        if args.do_cobevt:
            print("###########################################################################")
            print("["+str(i+1)+"/"+str(args.number)+"] Running Detections on covbevt: ")
            print("###########################################################################")
            start_time=time.time()
            detector.infer(cobevtmodeldir,cobevtcsvfile,test,args.saveinterval)
            print("###########################################################################")
            cobevttime=(time.time() - start_time)
            print("["+str(i+1)+"/"+str(args.number)+"] Done covbevt in %.2f seconds!" % cobevttime)
        if args.do_attfuse:
            print("###########################################################################")
            print("["+str(i+1)+"/"+str(args.number)+"] Running Detections on attfuse: ")
            print("###########################################################################")
            start_time=time.time()
            detector.infer(attfusemodeldir,attfusecsvfile,test,args.saveinterval)
            
            print("###########################################################################")
            attfusetime=(time.time() - start_time)
            print("["+str(i+1)+"/"+str(args.number)+"] Done attfuse in %.2f seconds!" % attfusetime)
        if args.do_disco:
            print("###########################################################################")
            print("["+str(i+1)+"/"+str(args.number)+"] Running Detections on disconet: ")
            print("###########################################################################")
            start_time=time.time()
            detector.infer(discomodeldir,discocsvfile,test,args.saveinterval)
            print("###########################################################################")
            discotime=(time.time() - start_time)
            print("["+str(i+1)+"/"+str(args.number)+"] Done disconet in %.2f seconds!" % discotime)
        
        print("###########################################################################")
        print("###########################################################################")
        print("Done spacing instance in %.2f seconds!" % (time.time() - looptime))
        print("Fcooper: %.2fs, Cobevt: %.2fs, Attfuse: %.2fs, Disconet: %.2fs" % (fcoopertime, cobevttime,attfusetime,discotime))
        print("Estimated time to completion, %.2f minutes, or %.2f hours!" % (float((time.time()-looptime)*(args.number-(i+1))/60), float((time.time()-looptime)*(args.number-(i+1))/3600)))
        print("###########################################################################")
        print("###########################################################################")

    print("###########################################################################")
    print("===END======END======END======END======END===")
    Ttime=(time.time() - TotalTimeStart)
    print("Done all in %.2f seconds!" % Ttime)
    print("That's %.2f minutes, or %.2f hours" % (Ttime/60, Ttime/3600))
    print("===END======END======END======END======END===")
    print("###########################################################################\n")

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('Done.')
