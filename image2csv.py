import cv2
import numpy as np
import csv

with open("image.csv", "a") as f:
    f.write("Image\n")
# 30x30 nomal

for i in range(7616):
    print((i/7616)*100,i)
    cnt = 0
    img = cv2.imread("PThermalALL/"+str(i).rjust(4, '0')+".jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.resize(img, dsize=(30, 30))
    img = cv2.copyMakeBorder(img, 33, 33, 33, 33, cv2.BORDER_CONSTANT, 0)
    data = img.reshape(-1)
    data = data.astype(np.uint8)
    with open("image.csv", "a") as f:
        f.write(" ".join(map(str, data)))
        

    with open("image.csv", "a") as f:
        f.write("\n")

# 30x30 bitwise

for i in range(7616):
    print((i/7616)*100,i)
    cnt = 0
    img = cv2.imread("PThermalALL/"+str(i).rjust(4, '0')+".jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.bitwise_not(img)
    img = cv2.resize(img, dsize=(30, 30))
    img = cv2.copyMakeBorder(img, 33, 33, 33, 33, cv2.BORDER_CONSTANT, 0)
    data = img.reshape(-1)
    data = data.astype(np.uint8)
    with open("image.csv", "a") as f:
        f.write(" ".join(map(str, data)))
        

    with open("image.csv", "a") as f:
        f.write("\n")


# 50x50 nomal

for i in range(7616):
    print((i/7616)*100,i)
    cnt = 0
    img = cv2.imread("PThermalALL/"+str(i).rjust(4, '0')+".jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.resize(img, dsize=(50, 50))
    img = cv2.copyMakeBorder(img, 23, 23, 23, 23, cv2.BORDER_CONSTANT, 0)
    data = img.reshape(-1)
    data = data.astype(np.uint8)
    with open("image.csv", "a") as f:
        f.write(" ".join(map(str, data)))
        

    with open("image.csv", "a") as f:
        f.write("\n")

#50x50 bitwise

for i in range(7616):
    print((i/7616)*100,i)
    cnt = 0
    img = cv2.imread("PThermalALL/"+str(i).rjust(4, '0')+".jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.bitwise_not(img)
    img = cv2.resize(img, dsize=(50, 50))
    img = cv2.copyMakeBorder(img, 23, 23, 23, 23, cv2.BORDER_CONSTANT, 0)
    data = img.reshape(-1)
    data = data.astype(np.uint8)
    with open("image.csv", "a") as f:
        f.write(" ".join(map(str, data)))
        

    with open("image.csv", "a") as f:
        f.write("\n")

# 70x70 nomal

for i in range(7616):
    print((i/7616)*100,i)
    cnt = 0
    img = cv2.imread("PThermalALL/"+str(i).rjust(4, '0')+".jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.resize(img, dsize=(70, 70))
    img = cv2.copyMakeBorder(img, 13, 13, 13, 13, cv2.BORDER_CONSTANT, 0)
    data = img.reshape(-1)
    data = data.astype(np.uint8)
    with open("image.csv", "a") as f:
        f.write(" ".join(map(str, data)))
        

    with open("image.csv", "a") as f:
        f.write("\n")


#70x70 bitwise

for i in range(7616):
    print((i/7616)*100,i)
    cnt = 0
    img = cv2.imread("PThermalALL/"+str(i).rjust(4, '0')+".jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.bitwise_not(img)
    img = cv2.resize(img, dsize=(70, 70))
    img = cv2.copyMakeBorder(img, 13, 13, 13, 13, cv2.BORDER_CONSTANT, 0)
    data = img.reshape(-1)
    data = data.astype(np.uint8)
    with open("image.csv", "a") as f:
        f.write(" ".join(map(str, data)))
        

    with open("image.csv", "a") as f:
        f.write("\n")

# 96x96 nomal

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

# 96x96 bitwise

for i in range(7616):
    print((i/7616)*100,i)
    cnt = 0
    img = cv2.imread("PThermalALL/"+str(i).rjust(4, '0')+".jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.bitwise_not(img)
    img = cv2.resize(img, dsize=(96, 96))
    data = img.reshape(-1)
    data = data.astype(np.uint8)
    with open("image.csv", "a") as f:
        f.write(" ".join(map(str, data)))
        

    with open("image.csv", "a") as f:
        f.write("\n")



# moving image


# left up

# 30x30 nomal

for i in range(7616):
    print((i/7616)*100,i)
    cnt = 0
    img = cv2.imread("PThermalALL/"+str(i).rjust(4, '0')+".jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.resize(img, dsize=(30, 30))
    img = cv2.copyMakeBorder(img, 0, 66, 0, 66, cv2.BORDER_CONSTANT, 0)
    data = img.reshape(-1)
    data = data.astype(np.uint8)
    with open("image.csv", "a") as f:
        f.write(" ".join(map(str, data)))
        

    with open("image.csv", "a") as f:
        f.write("\n")

#50x50 bitwise

for i in range(7616):
    print((i/7616)*100,i)
    cnt = 0
    img = cv2.imread("PThermalALL/"+str(i).rjust(4, '0')+".jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.bitwise_not(img)
    img = cv2.resize(img, dsize=(50, 50))
    img = cv2.copyMakeBorder(img, 0, 46, 0, 46, cv2.BORDER_CONSTANT, 0)
    data = img.reshape(-1)
    data = data.astype(np.uint8)
    with open("image.csv", "a") as f:
        f.write(" ".join(map(str, data)))
        

    with open("image.csv", "a") as f:
        f.write("\n")

# 70x70 nomal

for i in range(7616):
    print((i/7616)*100,i)
    cnt = 0
    img = cv2.imread("PThermalALL/"+str(i).rjust(4, '0')+".jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.resize(img, dsize=(70, 70))
    img = cv2.copyMakeBorder(img, 0, 26, 0, 26, cv2.BORDER_CONSTANT, 0)
    data = img.reshape(-1)
    data = data.astype(np.uint8)
    with open("image.csv", "a") as f:
        f.write(" ".join(map(str, data)))
        

    with open("image.csv", "a") as f:
        f.write("\n")





#left down

# 30x30 bitwise

for i in range(7616):
    print((i/7616)*100,i)
    cnt = 0
    img = cv2.imread("PThermalALL/"+str(i).rjust(4, '0')+".jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.bitwise_not(img)
    img = cv2.resize(img, dsize=(30, 30))
    img = cv2.copyMakeBorder(img, 66, 0, 0, 66, cv2.BORDER_CONSTANT, 0)
    data = img.reshape(-1)
    data = data.astype(np.uint8)
    with open("image.csv", "a") as f:
        f.write(" ".join(map(str, data)))
        

    with open("image.csv", "a") as f:
        f.write("\n")


# 50x50 nomal

for i in range(7616):
    print((i/7616)*100,i)
    cnt = 0
    img = cv2.imread("PThermalALL/"+str(i).rjust(4, '0')+".jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.resize(img, dsize=(50, 50))
    img = cv2.copyMakeBorder(img, 46, 0, 0, 46, cv2.BORDER_CONSTANT, 0)
    data = img.reshape(-1)
    data = data.astype(np.uint8)
    with open("image.csv", "a") as f:
        f.write(" ".join(map(str, data)))
        

    with open("image.csv", "a") as f:
        f.write("\n")


#70x70 bitwise

for i in range(7616):
    print((i/7616)*100,i)
    cnt = 0
    img = cv2.imread("PThermalALL/"+str(i).rjust(4, '0')+".jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.bitwise_not(img)
    img = cv2.resize(img, dsize=(70, 70))
    img = cv2.copyMakeBorder(img, 26, 0, 0, 26, cv2.BORDER_CONSTANT, 0)
    data = img.reshape(-1)
    data = data.astype(np.uint8)
    with open("image.csv", "a") as f:
        f.write(" ".join(map(str, data)))
        

    with open("image.csv", "a") as f:
        f.write("\n")

#right up

# 30x30 nomal

for i in range(7616):
    print((i/7616)*100,i)
    cnt = 0
    img = cv2.imread("PThermalALL/"+str(i).rjust(4, '0')+".jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.resize(img, dsize=(30, 30))
    img = cv2.copyMakeBorder(img, 0, 66, 66, 0, cv2.BORDER_CONSTANT, 0)
    data = img.reshape(-1)
    data = data.astype(np.uint8)
    with open("image.csv", "a") as f:
        f.write(" ".join(map(str, data)))
        

    with open("image.csv", "a") as f:
        f.write("\n")


#50x50 bitwise

for i in range(7616):
    print((i/7616)*100,i)
    cnt = 0
    img = cv2.imread("PThermalALL/"+str(i).rjust(4, '0')+".jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.bitwise_not(img)
    img = cv2.resize(img, dsize=(50, 50))
    img = cv2.copyMakeBorder(img, 0, 46, 46, 0, cv2.BORDER_CONSTANT, 0)
    data = img.reshape(-1)
    data = data.astype(np.uint8)
    with open("image.csv", "a") as f:
        f.write(" ".join(map(str, data)))
        

    with open("image.csv", "a") as f:
        f.write("\n")

# 70x70 nomal

for i in range(7616):
    print((i/7616)*100,i)
    cnt = 0
    img = cv2.imread("PThermalALL/"+str(i).rjust(4, '0')+".jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.resize(img, dsize=(70, 70))
    img = cv2.copyMakeBorder(img, 0, 26, 26, 0, cv2.BORDER_CONSTANT, 0)
    data = img.reshape(-1)
    data = data.astype(np.uint8)
    with open("image.csv", "a") as f:
        f.write(" ".join(map(str, data)))
        

    with open("image.csv", "a") as f:
        f.write("\n")




#right down



# 30x30 bitwise

for i in range(7616):
    print((i/7616)*100,i)
    cnt = 0
    img = cv2.imread("PThermalALL/"+str(i).rjust(4, '0')+".jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.bitwise_not(img)
    img = cv2.resize(img, dsize=(30, 30))
    img = cv2.copyMakeBorder(img, 66, 0, 66, 0, cv2.BORDER_CONSTANT, 0)
    data = img.reshape(-1)
    data = data.astype(np.uint8)
    with open("image.csv", "a") as f:
        f.write(" ".join(map(str, data)))
        

    with open("image.csv", "a") as f:
        f.write("\n")


# 50x50 nomal

for i in range(7616):
    print((i/7616)*100,i)
    cnt = 0
    img = cv2.imread("PThermalALL/"+str(i).rjust(4, '0')+".jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.resize(img, dsize=(50, 50))
    img = cv2.copyMakeBorder(img, 46, 0, 46, 0, cv2.BORDER_CONSTANT, 0)
    data = img.reshape(-1)
    data = data.astype(np.uint8)
    with open("image.csv", "a") as f:
        f.write(" ".join(map(str, data)))
        

    with open("image.csv", "a") as f:
        f.write("\n")


#70x70 bitwise

for i in range(7616):
    print((i/7616)*100,i)
    cnt = 0
    img = cv2.imread("PThermalALL/"+str(i).rjust(4, '0')+".jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    img = cv2.bitwise_not(img)
    img = cv2.resize(img, dsize=(70, 70))
    img = cv2.copyMakeBorder(img, 26, 0, 26, 0, cv2.BORDER_CONSTANT, 0)
    data = img.reshape(-1)
    data = data.astype(np.uint8)
    with open("image.csv", "a") as f:
        f.write(" ".join(map(str, data)))
        

    with open("image.csv", "a") as f:
        f.write("\n")