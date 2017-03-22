#!/usr/bin/env python
import cv2
import sys
import os


if __name__ == '__main__':
    datasetDir = sys.argv[1]
    outputDir = sys.argv[2]

    #output dataset directory
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    actions = os.listdir(datasetDir)
    for action in actions:
        print action
        actionDir = os.path.join(datasetDir, action)
        actionOutputDir = os.path.join(outputDir, action)

        #Action directory
        if not os.path.exists(actionOutputDir):
            os.mkdir(actionOutputDir)

        videos = os.listdir(actionDir)
        for video in videos:
            videoPath = os.path.join(actionDir, video)
            videoOutputDir = os.path.join(actionOutputDir, video)

            #Video directory
            if not os.path.exists(videoOutputDir):
                os.mkdir(videoOutputDir)

            frameNum = 0
            cam = cv2.VideoCapture(videoPath)
            success, image = cam.read()
            while success:
                success, image = cam.read()
                if success:
                    cv2.imwrite('{}\{}.jpg'.format(videoOutputDir, frameNum), image)
                    frameNum +=1