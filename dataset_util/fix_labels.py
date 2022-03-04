import os
import json



directory = "datasets/sample/imgs/all"



def fix_label(filename: str):
    data = None

    # Load json file
    with open(directory + "/" + filename, "r") as json_file:
        data = json.load(json_file)

    # Overwrite volo with volvo
    for shape in data["shapes"]:
        if shape["label"].startswith("volo"):
            shape["label"] = "volvo_part1"

    # Save to file        
    with open(directory + "/" + filename, "w") as json_file:
        json.dump(data, json_file, indent=4)



if __name__ == "__main__":
    fixed_files = 0
    for filename in os.listdir(directory):
        if filename.endswith(".json") and not filename.startswith(("all", "train", "val", "valid", "validate")):
            fix_label(filename)
        fixed_files += 1
    
    print(f"Files scanned: {fixed_files}")