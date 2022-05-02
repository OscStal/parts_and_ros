from typing import Any
from rotMatrix import quaternion_rotation_matrix

import yaml
import ruamel.yaml
import sys

def yaml_loader(path):
    with open(path) as stream:
        data = yaml.load(stream, Loader=yaml.FullLoader)
        a = data["phone"]["pose"]
        return a

def yaml_reader(x):
    x_str = str(x).zfill(10)
    return yaml_loader(r"C:\Users\caspe\Kandidat\data\Volvo\data\images\{0}_poses.yaml".format(x_str))

def yaml_writer():
    yaml = ruamel.yaml.YAML()
    #yaml.preserve_quotes = True
    with open(r"C:\Users\caspe\Kandidat\data\Volvo\gt.yml") as fp:
        data = yaml.load(fp)
    #x = 0
    for x in range(1020):
        for elem in data[x]:
            b = yaml_reader(x+1)
            for y in range(9):
                elem['cam_R_m2c'][y] = ruamel.yaml.scalarfloat.ScalarFloat(quaternion_rotation_matrix(b[1])[y], width=9,prec=1)
            for z in range(3):
                elem['cam_t_m2c'][z] = ruamel.yaml.scalarfloat.ScalarFloat(b[0][z]*1000, width=9, prec=1)
            elem['obj_id'] = 12
    with open(r"C:\Users\caspe\Kandidat\data\Volvo\gt.yml", "w") as file:
        yaml.dump(data, file)

yaml_writer()
