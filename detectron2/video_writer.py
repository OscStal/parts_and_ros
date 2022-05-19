import cv2
import numpy as np
import os, glob, re

if __name__ == "__main__":

    w, h = 480, 640
    vid_writer = cv2.VideoWriter("presentation/vids/newvid1.mp4", cv2.VideoWriter_fourcc("m", "p", "4", "v"), 10, (w, h))

    list_of_files = glob.glob("presentation/imgs/*")
    list_of_files.sort(key=lambda f: int(re.sub('\D', '', f)))

    for file in list_of_files:
        if "new" in file:
            print(file)
            file = cv2.imread(file)
            vid_writer.write(file)
    
    vid_writer.release()