import pandas as pd
import json
from math import sin, cos
import matplotlib.pylab as plt
import numpy as np
import os
import cv2
from collections import namedtuple
from glob import glob
from pathlib import Path

from carsmoney.utils import CAMERA as k

#build the dict
Label = namedtuple(
    'Label',
    [
        'name',  # The name of a car type
        'id',  # id for specific car type
        'category',  # The name of the car category, 'SUV', 'Sedan' etc
        'categoryId',  # The ID of car category. Used to create ground truth images
        # on category level.
    ])

models = [
    #     name   id   is_valid   category   categoryId
    Label('baojun-310-2017', 0, '2x', 0),
    Label('biaozhi-3008', 1, '2x', 0),
    Label('biaozhi-liangxiang', 2, '2x', 0),
    Label('bieke-yinglang-XT', 3, '2x', 0),
    Label('biyadi-2x-F0', 4, '2x', 0),
    Label('changanbenben', 5, '2x', 0),
    Label('dongfeng-DS5', 6, '2x', 0),
    Label('feiyate', 7, '2x', 0),
    Label('fengtian-liangxiang', 8, '2x', 0),
    Label('fengtian-MPV', 9, '2x', 0),
    Label('jilixiongmao-2015', 10, '2x', 0),
    Label('lingmu-aotuo-2009', 11, '2x', 0),
    Label('lingmu-swift', 12, '2x', 0),
    Label('lingmu-SX4-2012', 13, '2x', 0),
    Label('sikeda-jingrui', 14, '2x', 0),
    Label('fengtian-weichi-2006', 15, '3x', 1),
    Label('037-CAR02', 16, '3x', 1),
    Label('aodi-a6', 17, '3x', 1),
    Label('baoma-330', 18, '3x', 1),
    Label('baoma-530', 19, '3x', 1),
    Label('baoshijie-paoche', 20, '3x', 1),
    Label('bentian-fengfan', 21, '3x', 1),
    Label('biaozhi-408', 22, '3x', 1),
    Label('biaozhi-508', 23, '3x', 1),
    Label('bieke-kaiyue', 24, '3x', 1),
    Label('fute', 25, '3x', 1),
    Label('haima-3', 26, '3x', 1),
    Label('kaidilake-CTS', 27, '3x', 1),
    Label('leikesasi', 28, '3x', 1),
    Label('mazida-6-2015', 29, '3x', 1),
    Label('MG-GT-2015', 30, '3x', 1),
    Label('oubao', 31, '3x', 1),
    Label('qiya', 32, '3x', 1),
    Label('rongwei-750', 33, '3x', 1),
    Label('supai-2016', 34, '3x', 1),
    Label('xiandai-suonata', 35, '3x', 1),
    Label('yiqi-benteng-b50', 36, '3x', 1),
    Label('bieke', 37, '3x', 1),
    Label('biyadi-F3', 38, '3x', 1),
    Label('biyadi-qin', 39, '3x', 1),
    Label('dazhong', 40, '3x', 1),
    Label('dazhongmaiteng', 41, '3x', 1),
    Label('dihao-EV', 42, '3x', 1),
    Label('dongfeng-xuetielong-C6', 43, '3x', 1),
    Label('dongnan-V3-lingyue-2011', 44, '3x', 1),
    Label('dongfeng-yulong-naruijie', 45, 'SUV', 2),
    Label('019-SUV', 46, 'SUV', 2),
    Label('036-CAR01', 47, 'SUV', 2),
    Label('aodi-Q7-SUV', 48, 'SUV', 2),
    Label('baojun-510', 49, 'SUV', 2),
    Label('baoma-X5', 50, 'SUV', 2),
    Label('baoshijie-kayan', 51, 'SUV', 2),
    Label('beiqi-huansu-H3', 52, 'SUV', 2),
    Label('benchi-GLK-300', 53, 'SUV', 2),
    Label('benchi-ML500', 54, 'SUV', 2),
    Label('fengtian-puladuo-06', 55, 'SUV', 2),
    Label('fengtian-SUV-gai', 56, 'SUV', 2),
    Label('guangqi-chuanqi-GS4-2015', 57, 'SUV', 2),
    Label('jianghuai-ruifeng-S3', 58, 'SUV', 2),
    Label('jili-boyue', 59, 'SUV', 2),
    Label('jipu-3', 60, 'SUV', 2),
    Label('linken-SUV', 61, 'SUV', 2),
    Label('lufeng-X8', 62, 'SUV', 2),
    Label('qirui-ruihu', 63, 'SUV', 2),
    Label('rongwei-RX5', 64, 'SUV', 2),
    Label('sanling-oulande', 65, 'SUV', 2),
    Label('sikeda-SUV', 66, 'SUV', 2),
    Label('Skoda_Fabia-2011', 67, 'SUV', 2),
    Label('xiandai-i25-2016', 68, 'SUV', 2),
    Label('yingfeinidi-qx80', 69, 'SUV', 2),
    Label('yingfeinidi-SUV', 70, 'SUV', 2),
    Label('benchi-SUR', 71, 'SUV', 2),
    Label('biyadi-tang', 72, 'SUV', 2),
    Label('changan-CS35-2012', 73, 'SUV', 2),
    Label('changan-cs5', 74, 'SUV', 2),
    Label('changcheng-H6-2016', 75, 'SUV', 2),
    Label('dazhong-SUV', 76, 'SUV', 2),
    Label('dongfeng-fengguang-S560', 77, 'SUV', 2),
    Label('dongfeng-fengxing-SX6', 78, 'SUV', 2)
]
car_name2id = {label.name: label for label in models}
car_id2name = {label.id: label for label in models}


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


def load_data(data_path, train_path="train.csv"):
    data = pd.read_csv(os.path.join(data_path, train_path))
    # This may be the ugliest list comprehension I have ever written.
    data = pd.DataFrame(sum([[
        [tag, int(row[0]),
         np.array(list(map(float, row[1:])))]
        for row in np.reshape(batch, (-1, 7))
    ] for tag, batch in zip(data.ImageId, data.PredictionString.str.split(" "))
                             ], []),
                        columns=["ImageId", "CarId", "Prediction"])
    return data


def load_jsons(json_path="train/json"):
    jsons = {}
    for f in glob(f"{json_path}/*json"):
        name = Path(f).name.split(".")[0]
        with open(f) as json_file:
            jsons[name] = json.load(json_file)
    return jsons


def create_masks(data_path="train/data",
                 mask_path="train/data",
                 train_path="train.csv",
                 json_path="train/json", square_crop=True,
                 export_crops=True, export_masks=True):
    """Generates Segmenation masks from dataset provided a path."""
    if not (export_crops or export_masks):
        return
    train = load_data(data_path, train_path)
    jsons = load_jsons(json_path)
    for image_id, batch in train.groupby("ImageId"):
        image = cv2.imread(f"{data_path}/{image_id}.jpg",
                           cv2.COLOR_BGR2RGB)[:, :, ::-1]

        batch.reindex(
            np.linalg.norm(np.array([x for x in batch.Prediction])[:, -3:],
                           axis=1).argsort())

        overlay = np.zeros_like(image)
        for idx, car in batch.iterrows():
            data = jsons[car_id2name[car.CarId].name]

            # do the transformation from 3D to 2D projection
            yaw, pitch, roll, x, y, z = car.Prediction

            # pitch and yaw should be exchanged
            yaw, pitch, roll = -pitch, -yaw, -roll
            t = car.Prediction[-3:]

            # Get polyhedra information
            vertices = np.array(data['vertices'])
            vertices[:, 1] = -vertices[:, 1]
            triangles = np.array(data['faces']) - 1

            temp = np.zeros_like(image)
            Rt = np.eye(4)
            Rt[:3, 3] = t
            Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
            Rt = Rt[:3, :]
            P = np.ones((vertices.shape[0], vertices.shape[1] + 1))
            P[:, :-1] = vertices
            P = P.T
            img_cor_points = np.dot(k, np.dot(Rt, P))
            img_cor_points = img_cor_points.T
            img_cor_points[:, 0] /= img_cor_points[:, 2]
            img_cor_points[:, 1] /= img_cor_points[:, 2]
            draw_obj(temp, img_cor_points, triangles)

            # change it to boolean mask
            mask = (temp == 255)

            # Squarize and crop!
            if export_crops:
                left_corner = np.unravel_index(mask.argmax(), mask.shape)
                corner = np.unravel_index(np.flip(mask).argmax(), mask.shape)
                right_corner = (s - c for c, s in zip(corner, mask.shape))
                cropped = image[left_corner[0]:right_corner[0],
                        left_corner[1]:right_corner[1]]
                cv2.imwrite(f"{mask_path}/{image_id}_{idx}.jpg", cropped)
                plt.imshow(cropped)
                plt.show()
                break
            overlay[mask] = car.CarId

        if export_masks:
            cv2.imwrite(f"{mask_path}/{image_id}_mask.jpg", overlay[:, :, 2])
