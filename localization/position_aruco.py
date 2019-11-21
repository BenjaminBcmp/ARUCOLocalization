# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 16:28:06 2019

@author: romain
"""

import numpy as np
import cv2 as cv
from cv2 import aruco

# Tag number
ref = 2
robot = 3

markerLength = 0.06 # Length (in m) of the side of the tag

dictionary = aruco.Dictionary_get(aruco.DICT_4X4_100)

# Parameters webcam logitech
cameraMatrix = np.array([[1.405e+03, 0.00000000e+00, 6.211e+02],
                         [0.00000000e+00, 1.409e+03, 3.602e+02],
                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
distCoeffs = np.array([[-0.037,  0.46, -0.0025, -0.0064, -1.70]])

# Parameters webcam from Adrien laptop
# cameraMatrix = np.array([[960.4, 0.0, 641.5],
#                         [0.0, 963.1, 374.7],
#                         [0.0, 0.0, 1.0]])
# distCoeffs = np.array([[0.062, -0.39, 0.0033, -0.0051, 0.59]])

# resizeFactor = 1.0
# cv.resizeWindow('frame', int(1280*resizeFactor), int(720*resizeFactor))

# on laptop : 0 for laptop webcam, 1 for plugged usb webcam
# on raspberry pi : 0 for plugged usb webcam
cap = cv.VideoCapture(0)

# IMPORTANT NOTE : THE VIDEO STREAM RESOLUTION NEEDS TO BE THE SAME THAT THE
# IMAGES USED DURING THE CAMERA CALIBRATION PROCEDURE
# cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

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
        
        # Position of the robot with respect to tag ref
        if (ref in ids) and (robot in ids):
            # Translation and rotation of the robot and the ref
            rvecRef = rvecs[ids == ref]
            tvecRef = tvecs[ids == ref]
            rvecRobot = rvecs[ids == robot]
            tvecRobot = tvecs[ids == robot]
        
            rotRef, _ = cv.Rodrigues(rvecRef)
            rotRobot, _ = cv.Rodrigues(rvecRobot)
            
            # Matrix related to the robot
            transRobot = np.zeros((4,4))
            transRobot[0:3, 0:3] = rotRobot
            transRobot[0:3,3] = tvecRobot
            transRobot[3, 3] = 1
            
            # Matrix related to the ref
            rotRefT = rotRef.transpose()
            iTransRef = np.zeros((4,4))
            iTransRef[0:3, 0:3] = rotRefT
            iTransRef[0:3, 3] = (- rotRefT @ tvecRef.transpose()).flatten()
            iTransRef[3, 3] = 1
            
            # Position of the robot in the ref frame
            posRobotInRobot = np.array([0,0,0,1])
            posRobotInRef = iTransRef @ (transRobot @ posRobotInRobot)
            
            # Angle of the robot in the ref frame
            rotRobotInRef = np.zeros((3,3))
            rotRobotInRef = iTransRef @ transRobot
            angleRad = np.arctan2(rotRobotInRef[1,0], rotRobotInRef[0,0]) 
            angleDeg = angleRad*180/np.pi

            # Attention ne marche que si le tag aruco du robot reste parrallèle
            # au plan du sol (du tag de référence donc)
            print(angledeg)
        
    # If no marker is detected, we plot the camera stream
    else:
        cv.imshow("frame", imageBrute)
        
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()


