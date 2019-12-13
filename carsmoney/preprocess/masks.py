from math import sin, cos

import matplotlib.pylab as plt
import numpy as np
import cv2
import dask.dataframe as dd
from dask.multiprocessing import get

from carsmoney.preprocess.utils import load_data, load_jsons
from carsmoney.utils import CAMERA as k
from carsmoney.utils import ID2NAME


def get_extremes(image):
    top = np.unravel_index(image.argmax(), image.shape)
    corner = np.unravel_index(np.flip(image).argmax(), image.shape)
    bottom = tuple(s - c for c, s in zip(corner, image.shape))
    return (max(top[0] - 1, 1), max(bottom[0] - 1, 1))


def get_corners(image):
    extremes_row = get_extremes(image)
    extremes_col = get_extremes(np.rot90(image))
    return [
        extremes_row[0], extremes_row[1], -extremes_col[1], -extremes_col[0]
    ]


# convert euler angle to rotation matrix
def euler_to_Rot(yaw, pitch, roll):
    Y = np.array([[cos(yaw), 0, sin(yaw)], [0, 1, 0], [-sin(yaw), 0,
                                                       cos(yaw)]])
    P = np.array([[1, 0, 0], [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0], [sin(roll),
                                               cos(roll), 0], [0, 0, 1]])
    return np.dot(Y, np.dot(P, R))


def draw_obj(image, vertices, triangles):
    for t in triangles:
        coord = np.array(
            [vertices[t[0]][:2], vertices[t[1]][:2], vertices[t[2]][:2]],
            dtype=np.int32)
        cv2.fillConvexPoly(image, coord, (0, 0, 255))


def create_masks(data_path="train/data",
                 mask_path="train/data",
                 train_path="train.csv",
                 json_path="train/json",
                 square_crop=True,
                 export_crops=True,
                 mask_only=True,
                 export_masks=True,
                 partitions=30):
    """Generates Segmenation masks from dataset provided a path."""
    train = load_data(data_path, train_path)
    jsons = load_jsons(json_path)

    def format_image(image_id, image, overlay):
        def fn(car):
            idx = car.name
            data = jsons[ID2NAME[car.CarId].name]

            # do the transformation from 3D to 2D projection
            yaw, pitch, roll, x, y, z = car.Prediction

            # pitch and yaw should be exchanged
            yaw, pitch, roll = -pitch, -yaw, -roll

            # Get polyhedra information
            vertices = np.array(data['vertices'])
            vertices[:, 1] = -vertices[:, 1]
            triangles = np.array(data['faces']) - 1

            # Create rotation matrix
            Rt = np.eye(4)
            Rt[:3, 3] = car.Prediction[-3:]
            Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
            Rt = Rt[:3, :]

            # Concat 1s to vertices and flip
            P = np.ones((vertices.shape[0], vertices.shape[1] + 1))
            P[:, :-1] = vertices
            P = P.T

            # Apply rotation matrix to vertices
            img_cor_points = np.dot(k, np.dot(Rt, P))
            img_cor_points = img_cor_points.T
            img_cor_points[:, 0] /= img_cor_points[:, 2]
            img_cor_points[:, 1] /= img_cor_points[:, 2]

            mask = np.zeros_like(image)
            draw_obj(mask, img_cor_points, triangles)

            # change it to boolean mask
            mask = (mask == 255)

            # Squarize and crop!
            if export_crops:
                prefix = "crop"
                corners = get_corners(mask)
                cropped = image
                if mask_only:
                    prefix = "masked"
                    cropped[~mask] = 0
                cropped = cropped[corners[0]:corners[1], corners[2]:corners[3]]
                cv2.imwrite(f"{mask_path}/{prefix}_{image_id}_{idx}.jpg",
                            cropped)
            overlay[mask] = car.CarId

        return fn

    def process(batch):
        image_id = batch["ImageId"].iloc[0]
        image = cv2.imread(f"{data_path}/{image_id}.jpg",
                           cv2.COLOR_BGR2RGB)[:, :, ::-1]
        overlay = np.zeros_like(image)
        batch.set_index(batch.Prediction.apply(
            np.linalg.norm).argsort()).sort_index(ascending=False).apply(
                format_image(image_id, image, overlay), axis=1)
        if export_masks:
            cv2.imwrite(f"{mask_path}/{image_id}_mask.jpg", overlay[:, :, 2])
        return True

    dd.from_pandas(
        train, npartitions=partitions).groupby("ImageId").apply(
            process, meta=('processes', bool)).compute()
