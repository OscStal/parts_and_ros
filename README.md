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
- Detect occluded objects (partially visible objects)?
- Compute ideal grabbing point from bounding box and type of part
- Aruco somehow on the surrounding box
- Some 6D Object detection (EffecientPose?, Pix2Pose?)

