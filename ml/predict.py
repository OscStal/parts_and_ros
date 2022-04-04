import cv2, os, numpy, random, json
import torch
import detectron2

# Logger
from detectron2.utils.logger import setup_logger
# setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random, time

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg, CfgNode
from detectron2.utils.visualizer import Visualizer, ColorMode
import detectron2.data.transforms as T
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, Metadata
from detectron2 import model_zoo

from scipy import ndimage



def detect_objects():
    """
    Sets up config and model, creates Predictor, runs predictions on an image captured by camera and visualizes it
    """

    # Setup metadata and model
    register_coco_instances("train", {}, os.path.join(test_data_location, "a_json.json"), test_data_location)
    custom_metadata: Metadata = MetadataCatalog.get("train")

    # Config Setup, Inference should use the config with parameters that are used in training
    # Some of the config parameters might be unneccesary for testing on images
    cfg: CfgNode = get_cfg()
    setup_cfg(cfg)

    # Detection Setup
    print("Creating Predictor")
    predictor = DefaultPredictor(cfg)

    # Capture video source, 0 for webcam on laptop, might be 1 for external camera
    print("Adding video source")
    cap = cv2.VideoCapture(vidcap_id)

    if benchmark: time_list = []
    iters = 0

    while(True):
        print(f"Iteration: {iters}")
        print("Reading image")
        _, frame = cap.read()
        # frame = cv2.imread("test1.jpeg")

        print("Calculating outputs")
        if benchmark: t1 = time.perf_counter()
        outputs = predictor(frame) # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        if benchmark: t2 = time.perf_counter()

        shape = get_shape(outputs)
        print(f"Detected objects + image size: {shape}")

        visualize_all_objects(outputs, frame, custom_metadata)

        for mask in outputs['instances'].pred_masks:
            print(np.array(mask.shape))
            find_center(mask, frame)

        # Finds and plots a circle in the center of mass of the binary mask
        # cv2.waitKey(0) & 0xFF == ord('n')


        if benchmark: 
            tt1 = t2 - t1
            print(f"Time for prediction: {tt1}")
            time_list.append(tt1)
        iters += 1

        if (cv2.waitKey(1000) & 0xFF == ord('q')) or iters > 5:
            cap.release()
            cv2.destroyAllWindows()
            if benchmark:
                avg_time = (sum(time_list) / len(time_list))
                print(f"Average time: {avg_time}")
            break



def get_shape(outputs):
    if device == "cuda":
        shape = np.array(outputs['instances'].pred_masks.cpu()).shape
    if device == "cpu":
        shape = np.array(outputs['instances'].pred_masks).shape
    return shape


def visualize_all_objects(outputs, frame, custom_metadata):
    if visualize:
        v = Visualizer(frame[:, :, ::-1],
                    metadata=custom_metadata, 
                    scale=1.0, 
                    instance_mode=ColorMode.SEGMENTATION   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )

        # print("Creating drawable instance predictions")
        preds = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        pred_img = preds.get_image()[:, :, ::-1]

        print("Showing image")
        cv2.imshow("", pred_img)
        cv2.waitKey(10)


def setup_cfg(cfg):
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
    cfg.MODEL.DEVICE = device
    cfg.DATASETS.TRAIN = ("train", )
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1


def find_center(mask, image):
    """
    Finds the "center of mass" of the binary mask from a prediction
    """
    # numpy.set_printoptions(threshold=np.inf)  # Print entire binary mask

    if device == "cuda":
        mask = np.array(mask.cpu())
    if device == "cpu":
        mask = np.array(mask)
    # print(f"Mask is {mask}")

    center = ndimage.measurements.center_of_mass(mask)
    # center has coordinates in (y,x) and float, this makes it (x,y) and rounded integers
    center_int = tuple((int(x) for x in center))[::-1]
    # center_int_yx = tuple(int(x) for x in center)
    # center_int_xy = center_int_yx[::-1]
    print(f"Center pixel: {center_int}")

    circle_radius = 4
    circle_thickness = 2
    circle_color = (255, 0, 0)
    new_img = cv2.circle(image, center_int, circle_radius, circle_color, circle_thickness)
    cv2.imshow("Center of Mass", new_img)



vidcap_id: int = 1
model_path: str = "model_final.pth"
test_data_location = "datasets/sample/imgs/all"
device: str = "cpu"
benchmark: bool = False
visualize: bool = False

if __name__ == "__main__":
    detect_objects()