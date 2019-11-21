# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 16:28:06 2019

@author: adrie
"""

import numpy as np
import cv2 as cv
from cv2 import aruco


markerLength = 0.06 #Length (in m) of the side of the tag
idRef = 1 #Origin is defined by the aruco tag of id 1
path = 'C:/Adrien/Club robotique/vision par ordinateur/aruco/'

dictionary = aruco.Dictionary_get(aruco.DICT_4X4_100)

#Parameters webcam logitech
cameraMatrix = np.array([[1.405e+03, 0.00000000e+00, 6.211e+02],
                         [0.00000000e+00, 1.409e+03, 3.602e+02],
                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
distCoeffs = np.array([[-0.037,  0.46, -0.0025, -0.0064, -1.70]])

#Parameters webcam from Adrien laptop
#cameraMatrix = np.array([[960.4, 0.0, 641.5],
#                         [0.0, 963.1, 374.7],
#                         [0.0, 0.0, 1.0]])
#distCoeffs = np.array([[0.062, -0.39, 0.0033, -0.0051, 0.59]])

cv.namedWindow('frame', cv.WINDOW_NORMAL)
#resizeFactor = 1.0
#cv.resizeWindow('frame', int(1280*resizeFactor), int(720*resizeFactor))

#0 for laptop webcam, 1 for plugged usb webcam
cap = cv.VideoCapture(1)

#IMPORTANT NOTE : THE VIDEO STREAM RESOLUTION NEEDS TO BE THE SAME THAT THE
#IMAGES USED DURING THE CAMERA CALIBRATION PROCEDURE
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, imageBrute = cap.read()
    if not ret:
        continue
    
    corners, ids, rejectedImgPoints = aruco.detectMarkers(imageBrute, dictionary)
    # If a marker is detected
    if ids is not None and len(ids) > 0: 
        rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs)
        imageCorners = cv.aruco.drawDetectedMarkers(imageBrute, corners, ids)
        imageAxes = imageCorners.copy()
        for i in range(0, len(ids)):
            imageAxes = aruco.drawAxis(imageAxes, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], markerLength/2)
            
        cv.imshow("frame", imageAxes)
        
        # Position of tag number 4 (robot) with respect to tag number 1 (ref)
        # Ce code vous semble incompréhensible ? Allez voir romain :) 
        if (2 in ids) and (3 in ids):
            rvec2 = rvecs[ids == 2]
            tvec2 = tvecs[ids == 2]
            rvec3 = rvecs[ids == 3]
            tvec3 = tvecs[ids == 3]
            Rot2, _ = cv.Rodrigues(rvec2)
            Rot3, _ = cv.Rodrigues(rvec3)
            
            Trans3 = np.zeros((4,4))
            Trans3[0:3, 0:3] = Rot3
            Trans3[0:3,3] = tvec3
            Trans3[3, 3] = 1
            
            Rot2t = Rot2.transpose()
            ITrans2 = np.zeros((4,4))
            ITrans2[0:3, 0:3] = Rot2t
            ITrans2[0:3, 3] = (- Rot2t @ tvec2.transpose()).flatten()
            ITrans2[3, 3] = 1
            
            #Trans = ITrans1 @ Trans4
            Pos3in3 = np.array([0,0,0,1])
            Pos3in2 = ITrans2 @ (Trans3 @ Pos3in3)
            
            # Attention ne marche que si le tag aruco du robot reste parrallèle
            # au plan du sol (du tag de référence donc)
            Rot3in2 = np.zeros((3,3))
            Rot3in2 = ITrans2 @ Trans3
            anglerad = np.arctan2(Rot3in2[1,0], Rot3in2[0,0]) 
            angledeg = anglerad*180/np.pi
            print(angledeg)
            #Afficher position du tag 3
            #print(Pos3in2)
#        if (1 in ids):
#            tvec1 = tvecs[ids == 1]
#            print(tvec1)
        
    # If no marker is detected, we plot the camera stream
    else:
        cv.imshow("frame", imageBrute)
        
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()


