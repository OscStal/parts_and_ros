import yaml

def yaml_loader(path):
    with open(path) as stream:
        data = yaml.load(stream, Loader=yaml.FullLoader)
        a = data["phone"]["pose"]
        return a

def yaml_reader():
    for x in range(1, 1021):
        x_str = str(x).zfill(10)
        b = yaml_loader(r"C:\Users\caspe\Kandidat\data\Volvo\data\images\{0}_poses.yaml".format(x_str))

def yaml_writer():
    a = yaml_loader(r"C:\Users\caspe\Kandidat\data\Volvo\data\images\0000000001_poses.yaml")
    names_yaml = {
        'cam_R_m2c': {
            'cam_t_m2c': a[0],
            'obj_bb': 'user2',
            'obj_id': 12
        }
    }
    sdump = "- " + yaml.dump(names_yaml, indent=0)
    names = yaml.safe_load(sdump)

    with open(r"C:\Users\caspe\Kandidat\data\Volvo\data\gt.yml", "w") as f:
        yaml.safe_dump(names, f)

        #config = yaml.safe_load(f)
        #config[str(x-1)] = config['hostname'].copy()
        #config[str(x-1)]['rot'] = [b[0]]  # add the command as a list for the correct yaml
        #del config['hostname']  # del the 'hostname' key from config

   # with open('output.yml', 'a') as f:  # open the file in append mode
    #    yaml.dump(config, f, default_flow_style=False)
yaml_writer()