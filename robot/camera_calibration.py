import numpy as np
import cv2

# w, h = (1280, 720)

def calibrate_img(image, img_w, img_h):
    calibration_matrix = np.array([
            [1578.135315108312, 0.0, 625.6708621029746],
            [0.0, 1585.223944490997, 274.1438454056999],
            [0.0, 0.0, 1.0]
        ])
    # distortion_coefficients = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    distortion_coefficients = np.array([0.1913558390363024, 1.611580485047983, -0.0275432638538428, -0.0001706687576881858, -11.90379741245398])
    rectification_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    projection_matrix = np.array([[1578.135315108312, 0.0, 625.6708621029746, 0.0], [0.0, 1585.223944490997, 274.1438454056999, 0.0], [0.0, 0.0, 1.0, 0.0]])

    mapx, mapy = cv2.initUndistortRectifyMap(
            calibration_matrix, 
            distortion_coefficients, 
            rectification_matrix,
            projection_matrix,
            # new_cam_mtx,
            (img_w, img_h),
            cv2.CV_32FC1
        )
    return cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

if __name__ == "__main__":
    pass
