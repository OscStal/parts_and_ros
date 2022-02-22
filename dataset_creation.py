import cv2
import os


def imgs_from_video(file_path: str, file_prefix: str) -> None:
    video_capture = cv2.VideoCapture(file_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    rate: int = int(fps / image_capture_rate)
    success, image = video_capture.read()
    os.chdir(out_folder_path)
    count: int = 0
    while success:
        if (count % rate) == 0:
            cv2.imwrite(f"{file_prefix}{count//rate}.jpg", image)
            print(f"Saved image {count//rate} from frame {count}")
        success, image = video_capture.read()
        print(f"Created frame: {success}")
        count += 1



# Specify video path and output folder here
vid_file_path = r"datasets\linemod\vids\datasetfilm3.mp4"
out_folder_path = r"datasets\linemod\imgs"

# Also specify this, approximate amount of images captured per second of video
image_capture_rate: int = 2
image_filename_prefix = "datasetfilm3_"



if __name__ == "__main__":
    if not os.path.exists(out_folder_path):
        os.makedirs(out_folder_path)
    imgs_from_video(vid_file_path, image_filename_prefix)