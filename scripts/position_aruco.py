# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 16:28:06 2019

@author: adrie
"""

import numpy as np
import cv2 as cv
from cv2 import aruco
import Queue, threading, time

# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = Queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except Queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()


markerLength = 0.06 #Length (in m) of the side of the tag
idRef = 1 #Origin is defined by the aruco tag of id 1
path = 'C:/Adrien/Club robotique/vision par ordinateur/aruco/'

dictionary = aruco.Dictionary_get(aruco.DICT_4X4_100)

#Parameters webcam logitech
#cameraMatrix = np.array([[1.401e+03, 0.00000000e+00, 6.267e+02],
#                         [0.00000000e+00, 1.404e+03, 3.606e+02],
#                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
#distCoeffs = np.array([[-0.029,  0.31, -0.0029, -0.0057, -1.16]])

#Parameters webcam from Adrien laptop
cameraMatrix = np.array([[960.4, 0.0, 641.5],
                         [0.0, 963.1, 374.7],
                         [0.0, 0.0, 1.0]])
distCoeffs = np.array([[0.062, -0.39, 0.0033, -0.0051, 0.59]])

cv.namedWindow('frame', cv.WINDOW_NORMAL)
#resizeFactor = 1.0
#cv.resizeWindow('frame', int(1280*resizeFactor), int(720*resizeFactor))

#0 for laptop webcam, 1 for plugged usb webcam
#cap = cv.VideoCapture(0)
cap = VideoCapture(0)

#IMPORTANT NOTE : THE VIDEO STREAM RESOLUTION NEEDS TO BE THE SAME THAT THE
#IMAGES USED DURING THE CAMERA CALIBRATION PROCEDURE
# cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, imageBrute = cap.read()
    if not ret:
        continue
    
    corners, ids, rejectedImgPoints = aruco.detectMarkers(imageBrute, dictionary)
    #If a marker is detected
    if ids is not None and len(ids) > 0: 
        rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs)
        imageCorners = cv.aruco.drawDetectedMarkers(imageBrute, corners, ids)
        imageAxes = imageCorners.copy()
        for i in range(0, len(ids)):
            imageAxes = aruco.drawAxis(imageAxes, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], markerLength/2)
            
        cv.imshow("frame", imageAxes)
        
        #Position of tag number 4 (robot) with respect to tag number 1 (ref)
        if (1 in ids) and (4 in ids):
            rvec1 = rvecs[ids == 1]
            tvec1 = tvecs[ids == 1]
            rvec4 = rvecs[ids == 4]
            tvec4 = tvecs[ids == 4]
            Rot1, _ = cv.Rodrigues(rvec1)
            Rot4, _ = cv.Rodrigues(rvec4)
            
            Trans4 = np.zeros((4,4))
            Trans4[0:3, 0:3] = Rot4
            Trans4[0:3,3] = tvec4
            Trans4[3, 3] = 1
            
            Rot1t = Rot1.transpose()
            ITrans1 = np.zeros((4,4))
            ITrans1[0:3, 0:3] = Rot1t
            ITrans1[0:3, 3] = (- Rot1t @ tvec1.transpose()).flatten()
            ITrans1[3, 3] = 1
            
            #Trans = ITrans1 @ Trans4
            Pos4in4 = np.array([0,0,0,1])
            Pos4in1 = ITrans1 @ (Trans4 @ Pos4in4)
            print(Pos4in1)
#        if (1 in ids):
#            tvec1 = tvecs[ids == 1]
#            print(tvec1)
        
    #If no marker is detected, we plot the camera stream
    else:
        cv.imshow("frame", imageBrute)
        
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()


