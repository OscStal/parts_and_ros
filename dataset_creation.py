import cv2
import os

def imgs_from_video(file_prefix: str) -> None:
    """
    Creates a set of images from a source video file.
    Set the file_choice_dialog_enabled flag to True for pop-up-window file choice, to False to manually specify file path.
    If the manually specified output folder does not exist it will be created.
    """

    if file_choice_dialog_enabled:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        in_file = filedialog.askopenfilename(title="Choose source video file")
        if in_file is "":
            root.destroy()
            return
        out_folder = filedialog.askdirectory(title="Choose output folder")
        if out_folder is "":
            root.destroy()
            return

    else:
        in_file = vid_file_path
        out_folder = out_folder_path
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

    video_capture = cv2.VideoCapture(in_file)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    rate: int = int(fps / image_capture_rate)
    success, image = video_capture.read()
    os.chdir(out_folder)
    count: int = 0
    while success:
        if (count % rate) == 0:
            cv2.imwrite(f"{file_prefix}{count//rate}.jpg", image)
            print(f"Saved image {count//rate} from frame {count}")
        success, image = video_capture.read()
        print(f"Created frame: {success}")
        count += 1


# Manually specify these if file choice dialog is not enabled
vid_file_path: str = "datasets/test/test.mp4"
out_folder_path: str = "datasets/test/imgs"
image_capture_rate: int = 2
image_filename_prefix: str = "datasetfilm3_"

# Specify this
file_choice_dialog_enabled: bool = True


if __name__ == "__main__":
    imgs_from_video(image_filename_prefix)