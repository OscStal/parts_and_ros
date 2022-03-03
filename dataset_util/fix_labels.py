import os
import json



directory = "datasets/xxx"



def fix_label(filename: str):
    data = json.load(filename)
    for shape in data["shapes"]:
        if "volo" in shape["label"]:
            shape["label"].replace("volo", "volvo")

if __name__ == "__main__":
    for filename in os.scandir(directory):
        if ".json" in filename:
            fix_label(filename)