import json
cameraPosPath = ''
cameraPosheight:int
pixelEqCm:int 
pictureWidth:int
pictureHeight:int
objectCenterY:int
objectCenterX:int

def loadJson():
    f = open(cameraPosPath)
    return json.load(f)

def updateJson(positions:dict):
    deltaX=objectCenterX-pictureWidth/2*pixelEqCm
    deltaY=objectCenterY-pictureHeight/2*pixelEqCm
    positions['x']= positions['x'] + deltaX
    positions['y'] = positions['y'] + deltaY
    return positions
    




def main():
    e = updateJson(loadJson())
    with open("ObjectPosition.json", "w") as outfile:
        json.dump(e, outfile)


if __name__ == "__main__":
    main()

