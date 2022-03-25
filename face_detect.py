import cv2 as cv

# 定义人脸检测函数
def detect(frame):
    # 使用基于haar特征的检测器，要先将图片转为灰度图，加速计算特征
    frame_gray= cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    frame_gray = cv.equalizeHist(frame_gray)

    # 通过detectMultiScale多目标人脸检测来检测每一帧中出现的人脸，并返回包含人脸位置的元组列表
    faceRects = face_classifier.detectMultiScale(frame_gray, scaleFactor=1.2,minNeighbors=3,minSize=(32,32))

    # 遍历读取每一个人脸的位置信息
    for faceRect in faceRects:
        # 接受坐标位置信息
        x,y,w,h = faceRect
        # cv.rectangle()在指定图像绘制矩形，绘制人脸检测框
        frame = cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
    # 展示
    cv.imshow('face_detect',frame)



# 创建haar分类器对象
face_classifier = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
# 加载检测器文件
face_classifier.load("haarcascade_frontalface_default.xml")

# 打开笔记本内置摄像头（参数0）
video = cv.VideoCapture(0)

# 摄像头是否正常打开
if not video.isOpened:
    print("Error")
    exit(0)
while True:
    # 调用摄像头对象的read()方法，循环读取每一帧
    ret,frame = video.read()
    # 检测读取的帧数是否正常
    if frame is None:
        print("Error")
        break
    # 人脸检测函数
    detect(frame)
    #设置等待时间
    if cv.waitKey(10) == 27:
        break
