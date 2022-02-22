import cv2
import os


def imgs_from_video(file_path: str):
    video_capture = cv2.VideoCapture(file_path)
    success, image = video_capture.read()
    os.chdir(out_folder_path)
    count = 0
    while success:
        if (count % 10) == 0:
            cv2.imwrite(f"image{count//10}.jpg", image)
            print(f"Saved image {count//10} from frame {count}")
        success, image = video_capture.read()
        print(f"Created frame: {success}")
        count += 1

# Specify video path and output folder here
vid_file_path = r"datasets\one\vids\4_flat.mp4"
out_folder_path = r"datasets\one\imgs"

if __name__ == "__main__":
    if not os.path.exists(out_folder_path):
        os.makedirs(out_folder_path)
    imgs_from_video(vid_file_path)