import cv2
import numpy as np
import keras
from keras.models import Sequential, load_model


# 水平方向投影
def hProject(binary):
    h, w = binary.shape

    # 水平投影
    hprojection = np.zeros(binary.shape, dtype=np.uint8)

    # 创建h长度都为0的数组
    h_h = [0]*h
    for j in range(h):
        for i in range(w):
            if binary[j,i] == 0:
                h_h[j] += 1
    # 画出投影图
    for j in range(h):
        for i in range(h_h[j]):
            hprojection[j,i] = 255

    # cv2.imshow('hpro', hprojection)

    return h_h

# 垂直反向投影
def vProject(binary):
    h, w = binary.shape
    # 垂直投影
    vprojection = np.zeros(binary.shape, dtype=np.uint8)

    # 创建 w 长度都为0的数组
    w_w = [0]*w
    for i in range(w):
        for j in range(h):
            if binary[j, i ] == 0:
                w_w[i] += 1

    for i in range(w):
        for j in range(w_w[i]):
            vprojection[j,i] = 255

    # cv2.imshow('vpro', vprojection)

    return w_w


def load_model_cnn(model, all_imagea):
    predictt = []
    for imgf in all_imagea:
        # img = cv2.imread(f"{i}", 0)
        img = cv2.resize(imgf, (28, 28))
        img = 255 - img
        img = img.astype("float32")
        img_4 = img - np.amin(img)
        img_5 = 255 * img_4 / (np.amax(img_4))
        x_test_img = np.reshape(img_5, (1, 28, 28))
        x_Test4D = x_test_img.reshape(x_test_img.shape[0], 28, 28, 1).astype('float32')
        x_Test4D_normalize = (x_Test4D / np.amax(x_test_img))
        prediction = model.predict_classes(x_Test4D_normalize)
        predictt.append(prediction)
        # .append(prediction)
    return predictt


def load_model_cnn_unit(model, image):
    img = cv2.resize(image, (28, 28))
    img = 255 - img
    img = img.astype("float32")
    img_4 = img - np.amin(img)
    img_5 = 255 * img_4 / (np.amax(img_4))
    x_test_img = np.reshape(img_5, (1, 28, 28))
    x_Test4D = x_test_img.reshape(x_test_img.shape[0], 28, 28, 1).astype('float32')
    x_Test4D_normalize = (x_Test4D / np.amax(x_test_img))
    prediction = model.predict_classes(x_Test4D_normalize)
    return prediction

def trackChaned(x):
  pass

if __name__ == '__main__':
    all_image = []
    video = cv2.VideoCapture(1)
    model = load_model('my_model.h5')
    model.load_weights('my_model_weights.h5')

    # video.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    # frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # avi = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter("test123.mov", avi, 25, (1200, 600))
    lower_blue = np.array([78, 43, 46])
    upper_blue = np.array([110, 255, 255])
    
    cv2.namedWindow('Mask')
    cv2.createTrackbar("Min", "Mask", 0, 255, trackChaned)
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    tmp = None
    # cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    while True:
        all_imagea = []
        # Read a new frame
        ok, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        huh = cv2.getTrackbarPos("Min", "Mask")
        ret, th = cv2.threshold(gray, huh, 255, 0)
        cv2.imshow('originla', frame)
        cv2.imshow('Mask', th)
        
        # Exit if ESC pressed
        k = cv2.waitKey(100) & 0xff
        if k == 27:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            break
        if k == ord('p'):
            print("press the key == p")
            # framee = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # ret, th = cv2.threshold(framee, 127, 255, 0)
            h, w = gray.shape
            h_h = hProject(th)

            start = 0
            h_start, h_end = [], []
            position = []

            # 根据水平投影获取垂直分割
            for i in range(len(h_h)):
                if h_h[i] > 0 and start == 0:
                    h_start.append(i)
                    start = 1
                if h_h[i] == 0 and start == 1:
                    h_end.append(i)
                    start = 0

            for i in range(len(h_start)):
                cropImg = th[h_start[i]:h_end[i], 0:w]
                if i == 0:
                    pass
                    # cv2.imshow('cropimg', cropImg)
                    # cv2.imwrite('words_cropimg.jpg', cropImg)
                w_w = vProject(cropImg)

                wstart , wend, w_start, w_end = 0, 0, 0, 0
                for j in range(len(w_w)):
                    if w_w[j] > 0 and wstart == 0:
                        w_start = j
                        wstart = 1
                        wend = 0
                    if w_w[j] ==0 and wstart == 1:
                        w_end = j
                        wstart = 0
                        wend = 1

                    # 当确认了起点和终点之后保存坐标   
                    if wend == 1:
                        position.append([w_start, h_start[i], w_end, h_end[i]])
                        wend = 0

                # 确定分割位置
                for i, p in enumerate(position):
                    height = abs(p[1] - p[3])
                    weidgh = abs(p[0] - p[2])
                    y = height if height > weidgh else weidgh
                    x = height if height > weidgh else weidgh
                    # print(p)
                    # int((y - height) / 2)
                    center_point = (int((p[0] + p[2]) / 2), int((p[1] + p[3]) / 2))
                    print(f"center point {center_point}")
                    # print(int((center_point[0] - y / 2)), int((center_point[0] + y / 2)))
                    # print(int((center_point[0] - y / 2)), int((center_point[1] + x / 2)))
                    imgg = th[int((center_point[1] - y / 2)):int((center_point[1] + y / 2)),
                              int((center_point[0] - x / 2)):int((center_point[0] + x / 2))]
                    # print(int((y - height) / 2), int(y - (y - height) / 2))
                    # imgg = th[int((p[0] - height) / 2): int(p[0] - (p[0] - height) / 2),
                    #           int((p[1] - weidgh) / 2): int(p[1] - (p[1] - weidgh) / 2)]
                    imgg = cv2.resize(imgg, (28, 28))
                    # cv2.imwrite(f"{i}.jpg", imgg)
                    cv2.imshow(f"{i}", imgg)
                    # all_image.append(imgg)
                    predic = load_model_cnn_unit(model, imgg)
                    print(predic[0])
                    cv2.putText(frame, f"{predic[0]}", (center_point[0], center_point[1] - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (int((center_point[0] - x / 2)), int((center_point[1] - y / 2))),
                                  (int((center_point[0] + x / 2)), int((center_point[1] + y / 2))), (0, 0, 255), 2)
            # print(load_model_cnn(model, all_image))
            # cv2.imshow("th", th)
            cv2.imshow("test", frame)

    video.release()
    cv2.destroyAllWindows() 

