import random
import cv2
import numpy as np

# original size: 320*240*3
HEIGHTS = 128
WIDTHS = 171
CROP_SIZE = 112

def preprocess_each_video(video_path):
    vidcap = cv2.VideoCapture(video_path)
    res = []
    whiel True:
        ret, frame = vidcap.read()
        if ret:
            frame= cv2.resize(frame, (WIDTHS, HEIGHTS),
                              interpolation=cv2.INTER_AREA).astype(np.float32)
            scale = float(CROP_SIZE) / float(HEIGHTS)
            frame = cv2.resize(frame, (int(WIDTHS*scale+1), CROP_SIZE)
            crop_x = int((frame.shape[0] - CROP_SIZE) / 2)
            crop_y = int((frame.shape[1] - CROP_SIZE) / 2)
            frame = frame[crop_x:crop_x+CROP_SIZE, crop_y:crop+CROP_SIZE, :]
            noise = np.random.randint(0, 50, (CROP_SIZE, CROP_SIZE, 3))
            frame -= noise
            res.append(frame)
        else:
        break
    vidcap.release()
    return res
