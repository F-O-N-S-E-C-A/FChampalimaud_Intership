# Tiago Fonseca - 2018 - FlyDetection - FChampalimaud

import cv2
import os
import sys
from scipy import ndimage
import csv
from progress.bar import Bar


def csvReader(csvPath):
    with open(csvPath) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        data = []
        flag = False;
        for row in readCSV:
            if len(row) != 0:

                if row[0] == "T" and row[1] != "OE":
                    flag = False
                if flag:
                    temp = []
                    temp.append(row[2])
                    temp.append(row[3])
                    data.append(temp)
                if row[0] == "T" and row[1] == "OE":
                    flag = True
    return data


def mkDir(frameDirPath, dirName):
    # os.makedirs(frameDirPath)
    path = frameDirPath + "/" + dirName
    try:
        os.makedirs(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

    return path


def extractFrame(videoPath, dirPath, data):
    videoClip = cv2.VideoCapture(videoPath)
    success = 1
    count = 0

    dirNameNoOvopositor = input("\nwithout ovopositor diretory name:")
    dirNameOvopositor = input("with ovopositor diretory name:")

    dirPathNoOvopositor = mkDir(dirPath, dirNameNoOvopositor)
    dirPathOvopositor = mkDir(dirPath, dirNameOvopositor)

    length = int(videoClip.get(cv2.CAP_PROP_FRAME_COUNT))
    bar = Bar('Processing', max=length)
    while success:

        # progress(count, length, status='Doing very long job')

        # print('__ frame ' + str(count) +' __')

        success, image = videoClip.read()
        # cv2.imshow("video", image)
        # cv2.waitKey(1)

        # time = videoClip.get(cv2.CAP_PROP_POS_MSEC)
        # print(time)

        if not success: break

        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert image to grayscale
        ret, thr = cv2.threshold(gray_img, 160, 255, cv2.THRESH_BINARY_INV)  # binary threshold --> thr

        # cv2.imshow("Threshold", thr)
        # cv2.waitKey(1)

        contours, hierarchy = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # contours,hierarchy = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            # print('area:  ' + str(area))
            if 1500 > area > 900:

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

                # cv2.imshow('rotated' + str(i), trimmed)

                ovopositor = False
                for row in data:
                    # print(row)
                    # print(str(count))
                    if int(row[0]) < count < int(row[1]):
                        cv2.imwrite(os.path.join(dirPathOvopositor,
                                                 'frame-' + str(count) + '-ovipositor' + '_fly-' + str(i) + '.jpg'),
                                    trimmed)
                        ovopositor = True
                if ovopositor == False:
                    cv2.imwrite(os.path.join(dirPathNoOvopositor, 'frame-' + str(count) + '_fly-' + str(i) + '.jpg'),
                                trimmed)
                    # print("in op == false")

        count += 1
        bar.next()
    bar.finish()


if __name__ == '__main__':
    # videoPath = "/Users/tiagofonseca/Desktop/programming/Python/FlyDetection/samples/video21_2017-05-18T10_40_09.avi"
    # videoPath = "/Users/tiagofonseca/Desktop/programming/Python/FlyDetection/samples/video21.mp4"
    # path = "/Users/tiagofonseca/Desktop/programming/Python/FlyDetection/flyExtractor/data"
    #csvPath = "/Users/tiagofonseca/Desktop/programming/Python/FlyDetection/samples/ovipository/MA_video21.csv"
    videoPath = sys.argv[1]  # 1st argument --> video path
    path = sys.argv[2]
    csvPath = sys.argv[3]
    data = csvReader(csvPath)

    print('data was successfully read')

    extractFrame(videoPath, path, data)
