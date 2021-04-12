import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
path_cur=os.path.dirname(os.path.abspath(__file__))

import argparse
import time
from pathlib import Path
import glob
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from .utils.datasets import  letterbox
from .models.experimental import attempt_load
from .utils.datasets import LoadStreams, LoadImages
from .utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from .utils.plots import plot_one_box
from .utils.torch_utils import select_device, load_classifier, time_synchronized
import numpy as np
class YOLOV5():
    def __init__(self,draw=True):
        self.device=torch.device("cuda")
        self.path_model=os.path.join(path_cur,"weights/best.pt")
        self.model=attempt_load(self.path_model, map_location="cuda")
        self.names = model.module.names if hasattr(self.model, 'module') else self.model.names
        print(self.names)
        self.img_size=320
        self.conf_thres=0.4
        self.iou_thres=0.4
    def detect(self,im0s,draw=False):
        img = letterbox(im0s.copy(), new_shape=self.img_size)[0]
        # print(img.shape)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img) 
        img = torch.from_numpy(img).to(self.device).float()
        img /= 255.0  
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self.model(img, augment=False)[0]
        box_detects=[]
        ims=[]
        classes=[]
        confs=[]
        cls_ids=[]
        box_draw=[]
        res=[]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=None)
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                for *x, conf, cls in reversed(det):
                    if(self.names[int(cls)]=="biker" or self.names[int(cls)]=="person" ):
                        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
                        ims.append(im0s[c1[1]:c2[1],c1[0]:c2[0]])
                        top=c1[1]
                        left=c1[0]
                        right=c2[0]
                        bottom=c2[1]
                        res.append([left, top, right, bottom, conf, conf.item(), cls])
                        # box_detects.append([(left+right)//2,(top+bottom)//2, right-left,bottom-top])
                        # box_detects.append([left,])
                        box_draw.append([left,top,right-left,bottom-top])
                        classes.append(self.names[int(cls)])
                        # confs.append([conf.item()])
                        cls_ids.append(int(cls))
        if(draw):
             img=im0s
             font = cv2.FONT_HERSHEY_SIMPLEX
             for box,lb in zip(box_draw,classes):
                  img =cv2.rectangle(img,(box[0],box[1]),(box[2]+box[0],box[3]+box[1]),(0,255,0),3,3)
                  #img=cv2.putText(img,lb,(box[0],box[1]),font,2,(255,0,0),1)  
             return  box_detects,confs,img
       
        return res #bbox_xywh, cls_conf, cls_ids
    
    
if __name__ == '__main__':

    detector=YOLOV5()
    for path in glob.glob("test/*.jpg"):

        img=cv2.imread(path)
        
        boxes,ims,classes,img=detector.detect(img)
        print(len(boxes))
        font = cv2.FONT_HERSHEY_SIMPLEX 
        for box,im,lb in zip(boxes,ims,classes):
            print(lb)
            img =cv2.rectangle(img,(box[0],box[1]),(box[2]+box[0],box[3]+box[1]),(0,255,0),3,3)
            img=cv2.putText(img,lb,(box[0],box[1]),font,2,(255,0,0),1)
#         cv2.imshow("image",cv2.resize(img,(500,500)))
        cv2.waitKey(0)
