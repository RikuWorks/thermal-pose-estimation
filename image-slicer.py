import cv2
import sys

video_path = "ikumiRGB.mp4"

# 動画ファイル読込
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    # 正常に読み込めたのかチェックする
    # 読み込めたらTrue、失敗ならFalse
    print("動画の読み込み失敗")
    sys.exit()

# width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# fps = cap.get(cv2.CAP_PROP_FPS)

# print("width:{}, height:{}, count:{}, fps:{}".format(width,height,count,fps))

digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

n= 4507
while True:
    # read()でフレーム画像が読み込めたかを示すbool、フレーム画像の配列ndarrayのタプル
    is_image,frame_img = cap.read()
    if is_image:
        # 画像を保存
        cv2.imwrite("RGBikumi/" + str(n).zfill(digit) + ".jpg" , frame_img)
    else:
        # フレーム画像が読込なかったら終了
        break
    n += 1

cap.release()
