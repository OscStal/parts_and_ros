import re
import torch
import detectron2

# Logger
from detectron2.utils.logger import setup_logger
# setup_logger()

# Import common libraries
import numpy as np
import os, json, cv2, random, time, glob

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg, CfgNode
from detectron2.utils.visualizer import Visualizer, ColorMode
import detectron2.data.transforms as T
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, Metadata
from detectron2 import model_zoo


class MonoColorVisualizer(Visualizer):
    def _jitter(self, color):
        return color

def setup_cfg() -> CfgNode:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_path
    # cfg.MODEL_WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75
    cfg.MODEL.DEVICE = DEVICE
    # cfg.DATASETS.TRAIN = ("train", )
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    return cfg

def main():
    custom_metadata: Metadata = MetadataCatalog.get("train")
    cfg = setup_cfg()
    predictor = DefaultPredictor(cfg)
    list_of_files = glob.glob(img_dir)
    list_of_files.sort(key=lambda f: int(re.sub('\D', '', f)))

    for i, img in enumerate(list_of_files):
        print(i)
        t1 = time.perf_counter()
        img = cv2.imread(img)
        outputs = predictor(img)

        v = MonoColorVisualizer(img[:, :, ::-1],
        metadata=custom_metadata, 
        scale=1.0, 
        instance_mode=ColorMode.SEGMENTATION   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )

        preds = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        pred_img = preds.get_image()[:, :, ::-1]

        cv2.imwrite(f"presentation/imgs/newvid1_{i}.png", pred_img)
        t2 = time.perf_counter() - t1
        print(f"Est time: {(len(list_of_files)-i)*t2} seconds")
        cv2.waitKey(1)


DEVICE: str = "cpu"
model_path: str = "model_final.pth"
img_dir = "presentation/imgs/*"
benchmark: bool = False
visualize: bool = False
calibrate: bool = False

if __name__ == "__main__":
    main()