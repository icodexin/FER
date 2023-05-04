from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from video_page import Ui_MainWindow
from keras import models

import numpy as np
import cv2
import sys

# 导入已训练好的模型
model = models.load_model('./FER_Model.h5')
# 表情标签
emotion_map = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
# 加载分类器
classifier = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")


class VideoPageWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(VideoPageWindow, self).__init__(parent)
        self.timer_camera = QTimer()
        self.timer_video = QTimer()
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.setupUi(self)
        self.slot_init()
        self.once = True

    def slot_init(self):
        self.timer_camera.timeout.connect(self.show_camera)
        self.timer_video.timeout.connect(self.show_video)
        self.useCameraButton.clicked.connect(self.slotCameraButton)
        self.useVideoButton.clicked.connect(self.slotVideoButton)
        self.useImageButton.clicked.connect(self.slotImageButton)


    def show_camera(self):
        flag, frame = self.cap.read()  # 捕获帧
        self.image = cv2.flip(frame, 1)  # 镜像翻转
        self.predict(self.image)
        show = cv2.resize(self.image, (480, 320))  # 重置大小
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 转换颜色空间
        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)  # 转换成QImage
        self.videoLabel.setPixmap(QPixmap.fromImage(showImage))  # 设置UI

    def show_video(self):
        flag, self.image = self.cap.read()  # 捕获帧
        if flag:
            self.predict(self.image)
            show = cv2.resize(self.image, (480, 320))  # 重置大小
            show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 转换颜色空间
            showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)  # 转换成QImage
            self.videoLabel.setPixmap(QPixmap.fromImage(showImage))  # 设置UI

    def predict(self, image):
        # 转换成灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 检测人脸区域
        faceRects = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        color = (0, 0, 255)

        if len(faceRects):  # 大于0则检测到人脸
            for faceRect in faceRects:
                x, y, w, h = faceRect
                # 框出人脸
                cv2.rectangle(image, (x, y), (x + h, y + w), color, 2)
                # 获取人脸源
                src = gray[y:y + w, x:x + h]
                # 缩放至48*48
                img = cv2.resize(src, (48, 48))
                # 归一化
                img = img / 255.
                # 扩展维度
                x_input = np.expand_dims(img, axis=0)
                x_input = np.array(x_input, dtype='float32').reshape(-1, 48, 48, 1)
                # 预测输出
                y_output = model.predict(x_input)
                percent = self.getPercent(y_output[0])
                self.angryLabel.setText('生气：{:.2%}'.format(percent[0]))
                self.disgustLabel.setText('厌恶：{:.2%}'.format(percent[1]))
                self.fearLabel.setText('害怕：{:.2%}'.format(percent[2]))
                self.happyLabel.setText('高兴：{:.2%}'.format(percent[3]))
                self.sadLabel.setText('悲伤：{:.2%}'.format(percent[4]))
                self.surpriseLabel.setText('惊讶：{:.2%}'.format(percent[5]))
                self.neutralLabel.setText('中立：{:.2%}'.format(percent[6]))
                output_class = np.argmax(y_output[0])
                cv2.putText(image, emotion_map[output_class], (x, y), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 250), 5)

    def slotCameraButton(self):
        if self.timer_camera.isActive() == False:
            self.CAM_NUM = 0
            self.openCamera()
        else:
            self.closeCamera()

    def slotVideoButton(self):
        self.CAM_NUM = QFileDialog.getOpenFileName(self, "选择文件", "./", "视频文件 (*.mp4)")
        if self.CAM_NUM[0]:
            self.CAM_NUM = self.CAM_NUM[0]
            self.openVideo()

    def slotImageButton(self):
        self.CAM_NUM = QFileDialog.getOpenFileName(self, "选择文件", "./", "图像文件 (*.jpg *.jpeg *.bmp *.png)")
        if self.CAM_NUM[0]:
            self.CAM_NUM = self.CAM_NUM[0]
            self.openImage()

    def openCamera(self):
        if self.timer_video.isActive():
            self.closeVideo()
        flag = self.cap.open(self.CAM_NUM)
        if flag == False:
            msg = QMessageBox.warning(self, u'Warning', u'请监测相机与电脑是否连接正确',
                                      buttons=QMessageBox.Ok,
                                      defaultButton=QMessageBox.Ok)
        else:
            self.timer_camera.start(30)
            self.useCameraButton.setText("关闭摄像头")

    def closeCamera(self):
        self.timer_camera.stop()
        self.cap.release()
        self.videoLabel.clear()
        self.videoLabel.setText("图像/视频画面")
        self.useCameraButton.setText("使用摄像头")

    def openVideo(self):
        if self.timer_camera.isActive():
            self.closeCamera()
        flag = self.cap.open(self.CAM_NUM)
        if flag == False:
            msg = QMessageBox.warning(self, u'Warning', u'请检查视频文件是否没有损坏',
                                      buttons=QMessageBox.Ok,
                                      defaultButton=QMessageBox.Ok)
        else:
            self.timer_video.start(30)


    def closeVideo(self):
        self.timer_video.stop()
        self.cap.release()

    def openImage(self):
        if self.timer_video.isActive():
            self.closeVideo()
        if self.timer_camera.isActive():
            self.closeCamera()
        self.image = cv2.imread(self.CAM_NUM)
        self.predict(self.image)
        show = cv2.resize(self.image, (np.int(320 * self.image.shape[1] / self.image.shape[0]), 320))  # 重置大小
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 转换颜色空间
        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)  # 转换成QImage
        self.videoLabel.setPixmap(QPixmap.fromImage(showImage))  # 设置UI

    def getPercent(self, arr):
        sum = 0
        for n in arr:
            sum += n
        ret = []
        for n in arr:
            ret.append(n / sum)
        return ret


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = VideoPageWindow()
    myWin.show()
    sys.exit(app.exec_())
