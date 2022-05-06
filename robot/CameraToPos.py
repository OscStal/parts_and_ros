import json
stage = 1
cameraPosPath = "src/een150-lab-template/src/case_suite/case_scenes/scenarios/scenario_3/camera_over_table.json"
cameraPosPathStageTwo = "src/een150-lab-template/src/case_suite/case_scenes/scenarios/scenario_3/ObjectPosition.json"
# pixelEqCm:float = 0.0006849315 #0.0431654676*0.01 #0.000407 #0.0431654676*0.01
pixelEqCm:float = 0.0006482975 # Number of meters per pixel, ROS uses meters for its coordinates
pictureWidth:int = 640
pictureHeight:int = 480

def loadJson():
    f = open(cameraPosPath if stage==1 else cameraPosPathStageTwo)
    content = json.load(f)
    f.close()
    return content

def loadCameraJson():
    f = open("test.json")
    return json.load(f)

def updateCenterofPart(pos:dict):
    objectCenterX = pos['centers'][0][0]
    objectCenterY = pos['centers'][0][1]
    return (objectCenterX, objectCenterY)

def updateJson(positions:dict):
    global stage
    (objectCenterX, objectCenterY) = updateCenterofPart(loadCameraJson())
    deltaX = (objectCenterX - pictureWidth/2) * pixelEqCm
    deltaY = (objectCenterY - pictureHeight/2) * pixelEqCm
    positions['transform']['translation']['x'] = positions['transform']['translation']['x'] - deltaX + 0.02
    positions['transform']['translation']['y'] = positions['transform']['translation']['y'] + deltaY
    positions['transform']['translation']['z'] = 0.05
    
    positions['transform']['rotation']['x'] = 0.0008447198606275185
    positions['transform']['rotation']['y'] = -0.00042421270847366616
    positions['transform']['rotation']['z'] = -0.5690145646174372
    positions['transform']['rotation']['w'] = 0.8223269007792238

    positions['child_frame'] = 'ObjectPosition' if stage==1 else "ObjectPosition2"
    return positions

def updateJson2(positions:dict):
    (objectCenterX, objectCenterY) = updateCenterofPart(loadCameraJson())
    deltaX = (objectCenterX - pictureWidth/2) * pixelEqCm
    deltaY = (objectCenterY - pictureHeight/2) * pixelEqCm
    positions['transform']['translation']['x'] += -deltaX
    positions['transform']['translation']['y'] += deltaY
    positions['transform']['translation']['z'] = loadJson()["transform"]["translation"]["z"]

    positions['child_frame'] = 'ObjectPosition' if stage==1 else "ObjectPosition2"
    return positions

def main():
    lower = True
    global stage
    out_file = "src/een150-lab-template/src/case_suite/case_scenes/scenarios/scenario_3/ObjectPosition.json"
    out_file2 = "src/een150-lab-template/src/case_suite/case_scenes/scenarios/scenario_3/ObjectPosition2.json"
    e = updateJson(loadJson()) if lower else updateJson2(loadJson())
    with open(out_file if stage==1 else out_file2, "w") as outfile:
        json.dump(e, outfile, indent=4)


if __name__ == "__main__":
    main()

