# -*- coding: utf-8 -*-
"""
Created on Sun May 26 00:53:46 2019

@author: peng.zhou
"""
'''
这个版本的效果是没有选择box的功能
'''

#coding:utf-8
from flask import request, Flask
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import dlib
import cv2
import time
#import time
#import os
'''
下面方法会让客户端找不到URL
'''
#app = Flask(__name__)
#
#@app.route("/<name>", methods=['POST'])
#def get_frame(name):
#    print("enter the get_frame")
#    if name==1:
#        print ("service received 1--from kehuduan")
#        return 'service received 1--from service'
#    elif name==2:
#        print ("service received 2--from kehuduan")
#        return 'service received 2--from service'
#    else:
#        return "you input the wrong count"
#
#
#if __name__ == "__main__":
#    print("y1")
#    app.run("192.168.199.244", port=8080)
#    print("y2")
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-v", "--video", required=True,
    help="path to input video file")
ap.add_argument("-l", "--label", required=True,
    help="class label we are interested in detecting + tracking")
ap.add_argument("-o", "--output", type=str,
    help="path to optional output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])



app = Flask(__name__)

@app.route("/1/", methods=['POST'])
def get_1():
    print("[INFO] starting video stream...")
#    time.sleep(3)
    vs = cv2.VideoCapture(0)
    #参数传入数字时，代表打开摄像头，传入本地视频路径时，表示播放本地视频
    global tracker
    global writer
    global startX,startY,endX,endY
    tracker=None
    writer=None
    label = ""
    _,frame = vs.read()#fps:开始记录第一帧
    frame = imutils.resize(frame, width=800)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5,False)
    net.setInput(blob)
    detections = net.forward()
    print("detections is :\n",detections)
    print("type detections is:",type(detections))
#    cus_info = sorted(detections,key=lambda cus:cus[2],reverse=True)
#    detections[0, 0, :, 2].sort()
    
#    print("the sort_det is: ",cus_info)    
    
    '''
    检测模块
    '''
    s=[]
    print(type(s))
    # loop over the detections
    for i in np.arange(0, detections.shape[2]):#画出第一帧的目标检测结果
        confidence = detections[0, 0, i, 2]
        label_1="box{}".format(i)
        if confidence > 0.9 :
            idx = int(detections[0, 0, i, 1])
        
#        box = detections[0, 0, i, 3:7] * np.array([w*(2), h, w*(3/4), h*(3/4)])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
        
            label = "{}: {:.2f}%".format(CLASSES[idx],
                confidence * 100)
            
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                COLORS[idx], 2)
        
            y = startY - 15 if startY - 15 > 15 else startY + 15
        
            print("the startX, y is:\n",(startX, startY), (endX, endY))
        
            cv2.putText(frame, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            cv2.putText(frame, label_1, (startX+30, y+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            s.append([startX,startY,endX, endY])
            print("s1:",s)
    cv2.imshow("Frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    vs.release()
    print("Now have {} boxes!".format(len(s)))
#    dssw=input("input the box...:\n")
#    dssw=int(dssw)
#    if dssw==0:
#        startX,startY,endX,endY=s[0]
#        print("you choose the box0")
#    elif dssw==1:
#        startX,startY,endX,endY=s[1]
#        print("you choose the box1")
#    else: 
#        print("you print a wrong number!!!")
    return str(s)
'''
检测结束
'''
@app.route("/2/", methods=['POST'])
def get_2():
    global tracker
    global writer 
    global startX,startY,endX,endY
    vs = cv2.VideoCapture(0)
    fps = FPS().start()
    while True:
        (grabbed, frame) = vs.read()
        if frame is None:
            break
        frame = imutils.resize(frame, width=800)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#这一步是不是多余的
        if args["output"] is not None and writer is None:
            '''
            fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v')#python2.x
            # fourcc = cv2.VideoWriter_fourcc(*'MJPG') #python3.x
            '''
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                     (frame.shape[1], frame.shape[0]), True)#30为控制输出视频文件的播放速率
        if tracker is None:
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
#            rect = dlib.rectangle(int(startX), int(yz), int(endX), int(endY))
            '''
            start track:
                This object will start tracking the thing inside the bounding box in the given image. 
                That is, if you call update() with subsequent video frames then it will try to keep 
                track of the position of the object inside bounding_box.
                '''
            tracker.start_track(rgb, rect)
        else:
            '''
            DSST:简介
            https://blog.csdn.net/roamer_nuptgczx/article/details/50134633
            '''
            tracker.update(rgb)#update() for子序列
            pos = tracker.get_position()
        # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
        
        # draw the bounding box from the correlation object tracker
            cv2.rectangle(frame, (startX, startY), (endX, endY),
            (0, 255, 0), 2)
            cv2.putText(frame, "tracking", (startX, startY - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # check to see if we should write the frame to disk
        if writer is not None:
            writer.write(frame)

    # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # update the FPS counter
        fps.update()

# stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # check to see if we need to release the video writer pointer
    if writer is not None:
        writer.release()#按住q键停止的时候，释放写文件空间。

# do a bit of cleanup
    cv2.destroyAllWindows()
    vs.release()
    return "track over"
    
    
#    print("enter the get_2")
#    print ("service received 2--from kehuduan")
#    return 'service received 2--from service'


if __name__ == "__main__":
    print("y1")
    app.run("192.168.43.20", port=8080)
    print("y2")