import cv2
import sys
from copy import deepcopy
from imx_interface import CSI_Camera, gstreamer_pipeline

import dlib 

class Dlib_Tracker():
    def __init__(self, frame, init_box):
        self.__tracker = dlib.correlation_tracker()

        # Define an initial bounding box
        self.__rgb_init_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.__init_box = init_box.copy()

        # Parameters
        self.__threshold = 14.0

        # Init process
        self.__init_tracking()

    def __init_tracking(self):
        # Initialize tracker with first frame and bounding box
        self.__tracker.start_track(self.__rgb_init_frame, dlib.rectangle(int(self.__init_box[0]), int(self.__init_box[1]),\
                                                int(self.__init_box[2]), int(self.__init_box[3])))

    def process(self, frame, thres = 14):
        self.__threshold = thres
        
        # Read a new frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        score = self.__tracker.update(rgb_frame)
        ok = score > self.__threshold

        # Get the bounding box of current tracked object
        pos = self.__tracker.get_position()

        # unpack the position object
        startX = int(pos.left())
        startY = int(pos.top())
        endX = int(pos.right())
        endY = int(pos.bottom())

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Output
        if ok:
            # Tracking success
            bb = [startX, startY, endX, endY]
            res = True
        else :
            # Tracking failure
            self.__init_tracking()

            bb = [0,0,0,0]
            res = False

        return res, bb, score