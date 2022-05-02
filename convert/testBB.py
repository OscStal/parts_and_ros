from plyfile import PlyData
import numpy as np

def load_model_ply():
    """
   Loads a 3D model from a plyfile
    Args:
        path_to_ply_file: Path to the ply file containing the object's 3D model
    Returns:
        points_3d: numpy array with shape (num_3D_points, 3) containing the x-, y- and z-coordinates of all 3D model points

    """

    path_to_ply_file = "C:/Users/caspe/Kandidat/data/Volvo/models/silver_gun.ply"
    model_data = PlyData.read(path_to_ply_file)

    vertex = model_data['vertex']
    points_3d = np.stack([vertex[:]['x'], vertex[:]['y'], vertex[:]['z']], axis=-1)

    num_points = points_3d.shape[0]
    max_distances = np.zeros((num_points,))

    for i in range(num_points):
        point = points_3d[i, :]

        distances = np.linalg.norm(points_3d - point, axis=-1)
        max_distances[i] = np.max(distances)

    diameter = np.max(max_distances)

    print(diameter)

load_model_ply()