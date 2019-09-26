# Tiago Fonseca - 2018 - FlyDetection - FChampalimaud

import cv2
import os
import sys
from scipy import ndimage
from progress.bar import Bar


def mkDir(frameDirPath, dirName):
    # os.makedirs(frameDirPath)
    path = frameDirPath + "/" + dirName
    try:
        os.makedirs(path)
    except OSError:
        print("Creation of the directory %s failed" % frameDirPath)
    else:
        print("Successfully created the directory %s " % frameDirPath)

    return path


def extractFrame(videoPath, dirPath):
    videoClip = cv2.VideoCapture(videoPath)
    success = 1
    count = 0

    dirName = input("\ndirectory name:")
    dirPath = mkDir(dirPath, dirName)

    length = int(videoClip.get(cv2.CAP_PROP_FRAME_COUNT))
    bar = Bar('Processing', max=length)

    while success:

        #print('__ frame ' + str(count) +' __')

        success, image = videoClip.read()
        if not success: break

        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert image to grayscale
        ret, thr = cv2.threshold(gray_img, 160, 255, cv2.THRESH_BINARY_INV)  # binary threshold --> thr

        #cv2.imshow("Threshold", thr)
        # cv2.waitKey(1)

        contours, hierarchy = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # contours,hierarchy = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        count += 1
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            #print('area:  ' + str(area))
            if 2500 > area > 900:

                x, y, w, h = cv2.boundingRect(cnt)

                mX = mY = mXW = mYH = 90
                if x - mX < 0: mX = x
                if y - mY < 0: mY = y
                if x + w - mXW > gray_img.shape[1]: mXW = gray_img.shape[1] - x + w
                if y + h - mYH > gray_img.shape[0]: mYH = gray_img.shape[0] - y + h

                blobimg = gray_img[y - mY:y + h + mYH, x - mX:x + w + mXW]
                # cv2.imshow('blob-'+str(i), blobimg)

                (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
                rotated = ndimage.rotate(blobimg, angle)

                trimmed = rotated[mY - 5: rotated.shape[0] - mYH + 5, mX - 5: rotated.shape[1] - mXW + 5]

                #cv2.imshow('rotated' + str(i), trimmed)
                #cv2.waitKey(1)

                cv2.imwrite(os.path.join(dirPath, 'frame-' + str(count) + '_fly-' + str(i) + '.jpg'), trimmed)
        bar.next()
    bar.finish()

if __name__ == '__main__':
    #videoPath = "/Users/tiagofonseca/Desktop/programming/Python/FlyDetection/samples/video21_2017-05-18T10_40_09.avi"
    #videoPath = "/Users/tiagofonseca/Desktop/programming/Python/FlyDetection/samples/video21.mp4"
    #path = "/Users/tiagofonseca/Desktop/programming/Python/FlyDetection/flyExtractor/data"

    videoPath = sys.argv[1]  # 1st argument --> video path
    path = sys.argv[2]
    extractFrame(videoPath, path)
