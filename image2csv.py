import cv2
import numpy as np
import csv

with open("image.csv", "a") as f:
    f.write("Image\n")

for i in range(7616):
    print((i/7616)*100,i)
    cnt = 0
    img = cv2.imread("PThermalALL/"+str(i).rjust(4, '0')+".jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.resize(img, dsize=(96, 96))

    data = img.reshape(-1)
    data = data.astype(np.uint8)
    with open("image.csv", "a") as f:
        f.write(" ".join(map(str, data)))
        

    with open("image.csv", "a") as f:
        f.write("\n")

for k in range(17):
    for i in range(7616):
        print((i/7616)*100,i)
        cnt = 0
        img = cv2.imread("PThermalALL/"+str(i).rjust(4, '0')+".jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        height= img.shape[0]
        width= img.shape[1]
        center=(int(width/2),int(height/2))
        angle=20*(k+1)
        scale=1
        trans=cv2.getRotationMatrix2D(center,angle,scale)
        img=cv2.warpAffine(img,trans,(width,height))
        img = cv2.resize(img, dsize=(96, 96))

        data = img.reshape(-1)
        data = data.astype(np.uint8)
        with open("image.csv", "a") as f:
            f.write(" ".join(map(str, data)))
            

        with open("image.csv", "a") as f:
            f.write("\n")
for i in range(7616):
    print((i/7616)*100,i)
    cnt = 0
    img = cv2.imread("PThermalALL/"+str(i).rjust(4, '0')+".jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.resize(img, dsize=(96, 96))

    data = img.reshape(-1)
    data = data.astype(np.uint8)
    with open("image.csv", "a") as f:
        f.write(" ".join(map(str, data)))
        

    with open("image.csv", "a") as f:
        f.write("\n")

for k in range(17):
    for i in range(7616):
        print((i/7616)*100,i)
        cnt = 0
        img = cv2.imread("PThermalALL/"+str(i).rjust(4, '0')+".jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.bitwise_not(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        height= img.shape[0]
        width= img.shape[1]
        center=(int(width/2),int(height/2))
        angle=20*(k+1)
        scale=1
        trans=cv2.getRotationMatrix2D(center,angle,scale)
        img=cv2.warpAffine(img,trans,(width,height))
        img = cv2.resize(img, dsize=(96, 96))

        data = img.reshape(-1)
        data = data.astype(np.uint8)
        with open("image.csv", "a") as f:
            f.write(" ".join(map(str, data)))
            

        with open("image.csv", "a") as f:
            f.write("\n")
