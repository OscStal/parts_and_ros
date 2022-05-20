import cv2
import glob, re

if __name__ == "__main__":

    w, h = 480, 640
    vid_writer = cv2.VideoWriter("presentation/vids/detectron_viz3.mp4", cv2.VideoWriter_fourcc("m","p","4","v"), 15, (h, w))

    img_dir = "presentation/imgs1/*"

    list_of_files1 = glob.glob(img_dir)
    list_of_files1.sort(key=lambda f: int(re.sub('\D', '', f)))

    for file in list_of_files1:
        if "new" in file:
            print(file)
            file1 = cv2.imread(file)
            vid_writer.write(file1)
        else:
            pass
    
    vid_writer.release()