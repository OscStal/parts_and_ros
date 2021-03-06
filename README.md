# parts_and_ros
Repository for Bacherlor Thesis EENX15-22-23

# File structure
- "parts_and_ros"-folder
    - General filesS
    - "convert"-folder - things related to data-conversion for EfficientPose and related tools
    - "detectron2"-folder - everything detectron2 (running predictions, visualizations and finding center-points of objects, colab notebook)
    - "robot"-folder - everything related to sending and retrieving information to/from robot arm
    - "datasets"-folder - datasets used to train models
    - "dataset_util"-folder - scripts to generate data and convert data between formats

# Random (possibly) Useful Resources
All these resources were not used and not all resources used are neccesarily listed here
- 6D Object Detection with EfficientPose
    - [GitHub](https://github.com/ybkscht/EfficientPose)
- Facebook's Detectron2 using PyTorch (Removes empty space in bounding box, kind of)
    - [Intro](https://www.youtube.com/watch?v=1oq1Ye7dFqc)
    - [Short Machine Learning Use Case](https://www.youtube.com/watch?v=eUSgtfK4ivk)
    - [GitHub](https://github.com/facebookresearch/detectron2)
    - [Colab Template](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
    - [Useful Resource 1](https://gilberttanner.com/blog/detectron2-train-a-instance-segmentation-model)
    - [Detectron2 Explained[77min] - YouTube](https://www.youtube.com/watch?v=4woFgFM4PFU)
    - [Install gcc and g++ on Windows](https://www.youtube.com/watch?v=8CNRX1Bk5sY)
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
    - [ImgAug for augmentations (Looks most promising)](https://github.com/aleju/imgaug)

# Contributors
Erik B??ngsbo @ Chalmers

Casper Jarhult @ Chalmers

Jacob Nir @ Chalmers

Max Sedersten @ Chalmers

Oscar St??lnacke @ Chalmers

Ludvig Tyd??n @ Chalmers