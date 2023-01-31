import os
import random
import time
import pandas as pd
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.models import load_model
import sys
import tkinter
import cv2
import csv

fnum = 0

FTEST = 'temp.csv'
video_path = 'danceth.mp4'

# 映像読み込み
def videodataloader():
    with open(FTEST, "a") as f:
        f.write("Image\n")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("動画の読み込み失敗")
        sys.exit()

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    while True:
        is_image,frame_img = cap.read()
        if is_image:
                img = frame_img
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                img = clahe.apply(img)
                img = cv2.resize(img, dsize=(96, 96))
                data = img.reshape(-1)
                data = data.astype(np.uint8)
                with open(FTEST, "a") as f:
                    f.write(" ".join(map(str, data)))
                with open(FTEST, "a") as f:
                    f.write("\n")
                global fnum
                fnum+=1
        else:
            break

    cap.release()





def testdataloader():

    df = read_csv(os.path.expanduser(FTEST))
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    print(df.count())
    df = df.dropna()
    X = np.vstack(df['Image'].values) / 255.
    X = X.astype(np.float32)
    y = None
    return X, y



def keypoint_writer(y):
    temparx=y[0::2] * 720 + 720
    tempary=y[1::2] * 540 + 540
    xcnt=0
    ycnt=0
    with open("out.csv", "a") as f:
        for j in range (36):
            print(xcnt,ycnt)
            if(j%2==0):
                f.write(str(temparx[xcnt]))
                xcnt=xcnt+1
            elif(j%2==1):
                f.write(str(tempary[ycnt]))
                ycnt=ycnt+1
            if(j==35):
                f.write("\n")
            else:
                f.write(",")



def testdataloader2D():
    X, y = testdataloader()
    X = X.reshape(-1,96, 96,1)
    return X, y

def outputvideo():

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("動画の読み込み失敗")
        sys.exit()

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    framecounter=0
    data = pd.read_csv('out.csv', header=None)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter('video.mp4',fourcc, 30.0, (1440, 1080))
    while True:
        is_image,frame_img = cap.read()
        if is_image:
                for k in range(18):
                    cv2.circle(frame_img,(int(data.iat[framecounter,(k*2)]),int(data.iat[framecounter,(k*2)+1])),3,color=(255, 0, 0),thickness=-1)        
        else:
            break
        cv2.imwrite('a.jpg', frame_img)
        writer.write(frame_img)
        framecounter += 1
    writer.release()
    cap.release()




videodataloader()
model=load_model('model.h5')
X_test, _ = testdataloader2D()
y_test = model.predict(X_test)
for i in range(fnum):
    keypoint_writer(y_test[i])
outputvideo()