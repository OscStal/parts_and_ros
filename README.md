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
- Mask RCNN, similiar to Detectron
    - [GitHub](https://github.com/matterport/Mask_RCNN)
