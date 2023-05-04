import cv2 as cv
import numpy as np
from keras import models

model = models.load_model('./FER_Model.h5')

emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Can not open camera!")
    exit()

while True:
    # 逐帧捕获
    ret, frame = cap.read()
    frame = cv.flip(frame, 1)
    # 转换成灰度图像
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    classifier = cv.CascadeClassifier("./haarcascade_frontalface_default.xml")
    faceRects = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    color = (0, 0, 255)

    if len(faceRects):  # 大于0则检测到人脸
        for faceRect in faceRects:  # 单独框出每一张人脸
            x, y, w, h = faceRect
            # 框出人脸
            cv.rectangle(frame, (x, y), (x + h, y + w), color, 2)
            # 获取人脸源
            src = gray[y:y + w, x:x + h]
            # 缩放至48*48
            img = cv.resize(src, (48, 48))
            # 归一化
            img = img / 255.
            # 扩展维度
            x = np.expand_dims(img, axis=0)
            x = np.array(x, dtype='float32').reshape(-1, 48, 48, 1)
            # 预测输出
            y = model.predict(x)
            output_class = np.argmax(y[0])
            cv.putText(frame, emotion_map[output_class], (200, 100), cv.FONT_HERSHEY_COMPLEX,
                       2.0, (0, 0, 250), 5)
    cv.imshow("frame", frame)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
