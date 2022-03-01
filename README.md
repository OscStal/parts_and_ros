# parts_and_ros
Repository for Bacherlor Thesis EENX15-22-23

# File structure (Förslag)
- "requirements.txt", ".gitignore" and very general files in this top-level.

- Top Level - "parts_and_ros"-folder
    - "ml"-folder
        - Everything strictly related to the Machine Learning part of the project goes here

    - "ros"-folder
        - Everything strictly related to the ROS part of the project goes here

    - Dataset goes in top-level folder

# Ideas Machine Learning Part
- Normal Object Detection(YOLOv5?, more) on objects lying flat and include rotation as input somehow
    - Ratio between width and height of bounding box to approximate rotation?
    - Compute ideal grabbing point from bounding box and type of part
- Detect occluded objects (partially visible objects)?
- Do something with Aruco tags?
- 6D Object Detection with EfficientPose
    - [GitHub](https://github.com/ybkscht/EfficientPose)
- Facebook's Detectron2 using PyTorch (Removes empty space in bounding box, kind of)
    - [Intro](https://www.youtube.com/watch?v=1oq1Ye7dFqc)
    - [Short Machine Learning Use Case](https://www.youtube.com/watch?v=eUSgtfK4ivk)
    - [GitHub](https://github.com/facebookresearch/detectron2)
    - [Colab Template](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
    - [Useful Resource 1](https://gilberttanner.com/blog/detectron2-train-a-instance-segmentation-model)
    - [Detectron2 Explained[77min] - YouTube](https://www.youtube.com/watch?v=4woFgFM4PFU)
- Mask RCNN, similiar to Detectron
    - [GitHub](https://github.com/matterport/Mask_RCNN)
- Generating images with Blender
    - [BlenderProc](https://github.com/DLR-RM/BlenderProc)
    - [blAInder](https://github.com/ln-12/blainder-range-scanner)
    - [Blender For AI Developers Video "Course" (May or may not be useful)](https://www.immersivelimit.com/tutorials/blender-for-ai-developers)
        - [Some sort of Intro to this](https://www.immersivelimit.com/tutorials/synthetic-datasets-with-blender)
- Labeling
    - [Segments.ai](https://segments.ai/blog/speed-up-image-segmentation-with-model-assisted-labeling)
    - [LabelMe format to COCO format](https://github.com/Tony607/labelme2coco/blob/master/labelme2coco.py)
    - [Split coco JSON file into train.json and test.json](https://github.com/akarazniewicz/cocosplit)
    - [Possible data augmentation solution](https://github.com/joheras/CLoDSA)
    - [Data Augmentation from NVIDIA](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/use_cases/detection_pipeline.html)