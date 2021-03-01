import numpy as np
import cv2
from PIL import ImageGrab
import time

count = 0
while 1:
    count += 1
    img = ImageGrab.grab(bbox=(180, 150, 1800, 750)) #坐标
    img_np = np.array(img)
    frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    cv2.imshow('s', frame)

    if count == 5:
        t = time.time() * 1000
        cv2.imwrite('./data/' + str(int(t)) + '.png', frame)
        count = 0
        print('save')

    cv2.waitKey(5)
