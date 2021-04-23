import cv2
import numpy as np
import torch
from yolov5_original.detect import *
from motpy import Detection, MultiObjectTracker
from motpy.detector import BaseObjectDetector
from motpy.testing_viz import draw_detection, draw_track,draw_detection_box
from time import time
from PIL import Image, ImageDraw


FINAL_LINE_COLOR = (255, 255, 255)

class Counter:
    def __init__(self,polygon,url='town.avi'):
        self.detector = YOLOV5()
        model_spec = {
                'order_pos': 1, 'dim_pos': 2, # position is a center in 2D space; under constant velocity model
                'order_size': 0, 'dim_size': 2, # bounding box is 2 dimensional; under constant velocity model
                'q_var_pos': 1000., # process noise
                'r_var_pos': 0.1 # measurement noise
            }
        self.tracker = MultiObjectTracker(dt=1/30, model_spec=model_spec)
        self.url=url
        self.cam=cv2.VideoCapture(url)
        _,frame =self.cam.read()
        self.mark={}
        self.height,self.width = frame.shape[:2]
        #self.polygon=polygon+[(self.width,0),(0,0)]
        self.polygon=polygon
        self.counter_on=0
        self.counter_off=0
        self.create_mask()

    def create_mask(self):
        img = Image.new('L', (self.width,self.height), 0)
        ImageDraw.Draw(img).polygon(self.polygon, outline=1, fill=1)
        self.mask = np.array(img)
    def set_mask(self,polygon):
        self.polygon=polygon
        self.create_mask()

    def process_trackers(self,frame,tracks):

        for track in tracks:
            color=True
            if(len(track.trace)>1):

                x1,y1=track.trace[-2]
                x2,y2=track.trace[-1]
                if(self.mask[y1][x1]==False and self.mask[y2][x2]==True and (track.id not in self.mark.keys()) ):
                    self.mark[track.id]=1
                    self.counter_on+=1
                    color=False
                elif(self.mask[y1][x1]==True and self.mask[y2][x2]==False):
                    if(track.id in self.mark.keys()):
                        self.counter_on-=1
                        self.mark.pop(track.id)
                    else:
                        self.counter_off+=1
                        color=False
            # draw_detection_box(frame,track.box_cur)
            draw_track(frame, track,random_color=color)
    def put_res(self,frame):
        color = (255, 0, 0)
        frame = cv2.putText(frame, 'number of person on : '+str(self.counter_on), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, color, 2, cv2.LINE_AA)
        frame = cv2.putText(frame, 'number of person off : '+str(self.counter_off), (20,60), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, color, 2, cv2.LINE_AA)
        return frame
    def run(self):
        video =self.cam
        frame_num = 0
        ret, frame = video.read()
        height,width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter('output_23_4.avi',fourcc, 18, (width,frame_num))
        while 1:
            # try:
                print('--------------------------------')
                detection = []
                ret, frame = video.read()
                frame_num += 1
                if frame_num % 1 == 0:
                    start = time()
                    detections = self.detector.detect(frame)
                    for det in detections:
                        detection.append(Detection(box = np.array(det[:4])))
                        # draw_detection(frame, Detection(box = np.array(det[:4])))
                    
                    self.tracker.step(detections = detection)
                    tracks = self.tracker.active_tracks()
                    
                    
                    self.process_trackers(frame,tracks)
                    print("time : ",time()-start)
                    frame=self.put_res(frame)
                    frame=cv2.polylines(frame, np.array([self.polygon]), False, FINAL_LINE_COLOR, 1)
                    out_video.write(frame)
                   
                    cv2.imshow('frame', frame)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
            # except:
            #     pass
        out_video.release()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    X=Counter(polygon=[(13, 841), (1, 583), (117, 430), (116, 29), (777, 44), (1033, 138), (961, 473), (939, 855), (25, 841)],url="test.ts")
    X.run()
