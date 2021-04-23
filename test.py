from yolov5_original.detect import *
import cv2


img=cv2.imread("b2.jpeg")
detector = YOLOV5()

box_detects,confs,img  = detector.detect(img,draw=True)
cv2.imshow("image",img)
cv2.waitKey(0)

