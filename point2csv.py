import cv2
import numpy as np
import csv
import math

with open("train.csv", "a") as f:
    f.write("nose_x,nose_y,neck_x,neck_y,r_shoulder_x,r_shoulder_y,r_elbow_x,r_elbow_y,r_wrist_x,r_wrist_y,l_shoulder_x,l_shoulder_y,l_elbow_x,l_elbow_y,l_wrist_x,l_wrist_y,r_hip_x,r_hip_y,r_knee_x,r_knee_y,r_ankle_x,r_ankle_y,l_hip_x,l_hip_y,l_knee_x,l_knee_y,l_ankle_x,l_ankle_y,r_eye_x,r_eye_y,l_eye_x,l_eye_y,r_ear_x,r_ear_y,l_ear_x,l_ear_y\n")



#30x30
for i in range(2):
    for i in range(7616):
        imgsize=30
        print((i/7616)*100,i)
        cnt = 0
        with open("CSVALL/"+str(i).rjust(4, '0')+".csv") as d:
            reader =csv.reader(d)
            l = [row for row in reader]
            for j in range(18):
                try:
                    if(int(l[cnt][0])==j):
                        #print(int(l[cnt][0]))
                        with open("train.csv", "a") as f:
                            f.write(str((48-(imgsize/2))+((int(l[cnt][1]))/(1440/imgsize)))+","+str((48-(imgsize/2))+((int(l[cnt][2]))/(1080/imgsize)))+",")
                        cnt += 1
                    else:
                        with open("train.csv", "a") as f:
                            f.write(","+",")
                except:
                    if(j==17):
                        with open("train.csv", "a") as f:
                                f.write(","+",")
                    print("kesson")

        #with open("train.csv", "a") as f:
            #f.write(" ".join(map(str, data)))
            

        with open("train.csv", "a") as f:
            f.write("a\n")


#50x50
for i in range(2):
    for i in range(7616):
        imgsize=50
        print((i/7616)*100,i)
        cnt = 0
        with open("CSVALL/"+str(i).rjust(4, '0')+".csv") as d:
            reader =csv.reader(d)
            l = [row for row in reader]
            for j in range(18):
                try:
                    if(int(l[cnt][0])==j):
                        #print(int(l[cnt][0]))
                        with open("train.csv", "a") as f:
                            f.write(str((48-(imgsize/2))+((int(l[cnt][1]))/(1440/imgsize)))+","+str((48-(imgsize/2))+((int(l[cnt][2]))/(1080/imgsize)))+",")
                        cnt += 1
                    else:
                        with open("train.csv", "a") as f:
                            f.write(","+",")
                except:
                    if(j==17):
                        with open("train.csv", "a") as f:
                                f.write(","+",")
                    print("kesson")

        #with open("train.csv", "a") as f:
            #f.write(" ".join(map(str, data)))
            

        with open("train.csv", "a") as f:
            f.write("a\n")




#70x70
for i in range(2):
    for i in range(7616):
        imgsize=70
        print((i/7616)*100,i)
        cnt = 0
        with open("CSVALL/"+str(i).rjust(4, '0')+".csv") as d:
            reader =csv.reader(d)
            l = [row for row in reader]
            for j in range(18):
                try:
                    if(int(l[cnt][0])==j):
                        #print(int(l[cnt][0]))
                        with open("train.csv", "a") as f:
                            f.write(str((48-(imgsize/2))+((int(l[cnt][1]))/(1440/imgsize)))+","+str((48-(imgsize/2))+((int(l[cnt][2]))/(1080/imgsize)))+",")
                        cnt += 1
                    else:
                        with open("train.csv", "a") as f:
                            f.write(","+",")
                except:
                    if(j==17):
                        with open("train.csv", "a") as f:
                                f.write(","+",")
                    print("kesson")

        #with open("train.csv", "a") as f:
            #f.write(" ".join(map(str, data)))
            

        with open("train.csv", "a") as f:
            f.write("a\n")





#96x96
for i in range(2):
    for i in range(7616):
        print((i/7616)*100,i)
        cnt = 0
        with open("CSVALL/"+str(i).rjust(4, '0')+".csv") as d:
            reader =csv.reader(d)
            l = [row for row in reader]
            for j in range(18):
                try:
                    if(int(l[cnt][0])==j):
                        #print(int(l[cnt][0]))
                        with open("train.csv", "a") as f:
                            f.write(str((int(l[cnt][1]))/(1440/96))+","+str((int(l[cnt][2]))/(1080/96))+",")
                        cnt += 1
                    else:
                        with open("train.csv", "a") as f:
                            f.write(","+",")
                except:
                    if(j==17):
                        with open("train.csv", "a") as f:
                                f.write(","+",")
                    print("kesson")

        #with open("train.csv", "a") as f:
            #f.write(" ".join(map(str, data)))
            

        with open("train.csv", "a") as f:
            f.write("a\n")








