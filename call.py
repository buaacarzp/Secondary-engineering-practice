# -*- coding: utf-8 -*-
"""
Created on Sat May 25 02:01:48 2019

@author: peng.zhou
"""

from subprocess import call
#import os
zhoupeng=call(["python","./backup/Python_service.py","-p","./mobilenet_ssd/MobileNetSSD_deploy.prototxt","-m",
      "./mobilenet_ssd/MobileNetSSD_deploy.caffemodel","-v","./input/cat.mp4",
      "-l","person","-o","./output/cat_output.avi","-c","0.9"])
print("zhoupeng is:",zhoupeng)
