import torch
import detectron2

# Logger
from detectron2.utils.logger import setup_logger
# setup_logger()

# Import common libraries
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

    # Setup metadata, might not be neccessary
    # register_coco_instances("train", {}, os.path.join(test_data_location, "a_json.json"), test_data_location)
    custom_metadata: Metadata = MetadataCatalog.get("train")

    # Config Setup, Inference should use the config with parameters that are used in training
    # Some of the config parameters might be unneccesary for testing on images
    cfg = setup_cfg()

    # Detection Setup
    print("Creating Predictor...")
    predictor = DefaultPredictor(cfg)

    # Capture video source, 0 for webcam on laptop, might be 1 for external camera
    print("Adding video source...")
    cap = cv2.VideoCapture(VIDCAP_ID)

    # Set resolution
    cap.set(3, 1280)
    cap.set(4, 720)

    if benchmark: time_list = []
    iters = 0

    while(True):
        print(f"Iteration: {iters}")
        print("Reading image...")
        _, frame = cap.read()
        # frame = cv2.imread("test1.jpeg")

        print("Calculating outputs...")
        if benchmark: t1 = time.perf_counter()
        outputs = predictor(frame) # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        if benchmark: t2 = time.perf_counter()

        shape = get_mask_tensor_shape(outputs)
        print(f"Detected objects + image size: {shape}")

        if visualize:
            visualize_mask_and_center_all(outputs, frame, custom_metadata, iters)

        all_masks = outputs['instances'].pred_masks

        object_centers = list(find_center(mask, frame, iters) for mask in all_masks)
        img_width = shape[2]
        img_height = shape[1]
        pixel_center = (img_width/2, img_height/2)
        object_offsets = tuple(tuple(map(lambda i, j: i - j, object_center, pixel_center)) for object_center in object_centers)
        with open("test.json", "w") as file:
            json.dump({
                "img_dim": {"width": img_width, "height": img_height},
                "centers": object_centers,
                "center_offsets": object_offsets
                },
                file,
                indent = 4
                )

        # for mask in all_masks:
        #     # print(np.array(mask.shape))
        #     find_center(mask, frame)

        iters += 1

        if (True):
            cap.release()
            cv2.waitKey(0) & 0xFF == ord("q")
            cv2.destroyAllWindows()
            if benchmark:
                tt1 = t2 - t1
                print(f"Time for prediction: {tt1}")
                time_list.append(tt1)
                avg_time = (sum(time_list) / len(time_list))
                print(f"Average time: {avg_time}")
            break



def get_mask_tensor_shape(outputs):
    if DEVICE == "cuda":
        shape = np.array(outputs['instances'].pred_masks.cpu()).shape
    if DEVICE == "cpu":
        shape = np.array(outputs['instances'].pred_masks).shape
    return shape


def visualize_mask_and_center_all(outputs, frame, custom_metadata, iters):
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
    cv2.imwrite(f"full_img{iters}", pred_img)
    cv2.waitKey(10)
    return pred_img


def setup_cfg() -> CfgNode:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
    cfg.MODEL.DEVICE = DEVICE
    # cfg.DATASETS.TRAIN = ("train", )
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    return cfg


def find_center(mask, image, iters):
    """
    Finds the "center of mass" of the binary mask from a prediction
    """
    if DEVICE == "cuda":
        mask = np.array(mask.cpu())
    if DEVICE == "cpu":
        mask = np.array(mask)

    # numpy.set_printoptions(threshold=np.inf)  # Print entire binary mask
    # print(f"Mask is {mask}")
    center = ndimage.center_of_mass(mask)
    # center has coordinates in (y,x) and float, this makes it (x,y) and rounded integers
    center_int = tuple((int(x) for x in center))[::-1]
    print(f"Center pixel: {center_int}")

    CIRCLE_RADIUS = 4
    CIRCLE_THICKNESS = 2
    CIRCLE_COLOR = (255, 0, 0)
    if visualize:
        new_img = cv2.circle(image, center_int, CIRCLE_RADIUS, CIRCLE_COLOR, CIRCLE_THICKNESS)
        cv2.imwrite(f"center_img{iters}", new_img)
        cv2.imshow("Center of Mass", new_img)

    return center_int



VIDCAP_ID: int = 1
DEVICE: str = "cpu"
model_path: str = "model_final.pth"
test_data_location = "datasets/sample/imgs/all"
benchmark: bool = False
visualize: bool = True

if __name__ == "__main__":
    detect_objects()