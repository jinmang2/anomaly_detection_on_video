from tqdm import tqdm
import os
import cv2
from PIL import Image


def main(filepath = 'E:/UCF101/UCF-101'):
    avi_generater = [contents for contents in os.walk(filepath)]
    for parent, dirnames, filenames in tqdm(avi_generater):
        if len(filenames) == 0: continue
        for filename in filenames:
            if filename.split('.')[1] != 'avi': continue
            orig_filename = filename.split('.avi')[0]
            img_path = os.path.join(parent, orig_filename)
            if not os.path.exists(img_path):
                os.makedirs(img_path)
            vidcap = cv2.VideoCapture(os.path.join(parent, filename))
            num = 1
            while True:
                ret, frame = vidcap.read()
                if not ret: break
                im = Image.fromarray(frame)
                im.save(
                    os.path.join(
                        img_path, orig_filename+'{:05d}.jpg'.format(num)))
                num += 1
            
if __name__ == '__main__':
    main()
