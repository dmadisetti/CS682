import os
import numpy as np
from collections import namedtuple

PROVISIONING_SCRIPT = open(
    os.path.join(os.path.dirname(__file__), 'provision.sh')).read()

CAMERA = np.array(
    [[2304.5479, 0, 1686.2379], [0, 2305.8757, 1354.9849], [0, 0, 1]],
    dtype=np.float32)

# Static data on cars
CarInfo = namedtuple(
    'CarInfo',
    [
        'name',  # The name of a car type
        'id',  # id for specific car type
        'category',  # The name of the car category, 'SUV', 'Sedan' etc
        'categoryId',  # The ID of car category. Used to create ground truth images
        # on category level.
    ])

MODELS = [
    #     name   id   is_valid   category   categoryId
    CarInfo('baojun-310-2017', 0, '2x', 0),
    CarInfo('biaozhi-3008', 1, '2x', 0),
    CarInfo('biaozhi-liangxiang', 2, '2x', 0),
    CarInfo('bieke-yinglang-XT', 3, '2x', 0),
    CarInfo('biyadi-2x-F0', 4, '2x', 0),
    CarInfo('changanbenben', 5, '2x', 0),
    CarInfo('dongfeng-DS5', 6, '2x', 0),
    CarInfo('feiyate', 7, '2x', 0),
    CarInfo('fengtian-liangxiang', 8, '2x', 0),
    CarInfo('fengtian-MPV', 9, '2x', 0),
    CarInfo('jilixiongmao-2015', 10, '2x', 0),
    CarInfo('lingmu-aotuo-2009', 11, '2x', 0),
    CarInfo('lingmu-swift', 12, '2x', 0),
    CarInfo('lingmu-SX4-2012', 13, '2x', 0),
    CarInfo('sikeda-jingrui', 14, '2x', 0),
    CarInfo('fengtian-weichi-2006', 15, '3x', 1),
    CarInfo('037-CAR02', 16, '3x', 1),
    CarInfo('aodi-a6', 17, '3x', 1),
    CarInfo('baoma-330', 18, '3x', 1),
    CarInfo('baoma-530', 19, '3x', 1),
    CarInfo('baoshijie-paoche', 20, '3x', 1),
    CarInfo('bentian-fengfan', 21, '3x', 1),
    CarInfo('biaozhi-408', 22, '3x', 1),
    CarInfo('biaozhi-508', 23, '3x', 1),
    CarInfo('bieke-kaiyue', 24, '3x', 1),
    CarInfo('fute', 25, '3x', 1),
    CarInfo('haima-3', 26, '3x', 1),
    CarInfo('kaidilake-CTS', 27, '3x', 1),
    CarInfo('leikesasi', 28, '3x', 1),
    CarInfo('mazida-6-2015', 29, '3x', 1),
    CarInfo('MG-GT-2015', 30, '3x', 1),
    CarInfo('oubao', 31, '3x', 1),
    CarInfo('qiya', 32, '3x', 1),
    CarInfo('rongwei-750', 33, '3x', 1),
    CarInfo('supai-2016', 34, '3x', 1),
    CarInfo('xiandai-suonata', 35, '3x', 1),
    CarInfo('yiqi-benteng-b50', 36, '3x', 1),
    CarInfo('bieke', 37, '3x', 1),
    CarInfo('biyadi-F3', 38, '3x', 1),
    CarInfo('biyadi-qin', 39, '3x', 1),
    CarInfo('dazhong', 40, '3x', 1),
    CarInfo('dazhongmaiteng', 41, '3x', 1),
    CarInfo('dihao-EV', 42, '3x', 1),
    CarInfo('dongfeng-xuetielong-C6', 43, '3x', 1),
    CarInfo('dongnan-V3-lingyue-2011', 44, '3x', 1),
    CarInfo('dongfeng-yulong-naruijie', 45, 'SUV', 2),
    CarInfo('019-SUV', 46, 'SUV', 2),
    CarInfo('036-CAR01', 47, 'SUV', 2),
    CarInfo('aodi-Q7-SUV', 48, 'SUV', 2),
    CarInfo('baojun-510', 49, 'SUV', 2),
    CarInfo('baoma-X5', 50, 'SUV', 2),
    CarInfo('baoshijie-kayan', 51, 'SUV', 2),
    CarInfo('beiqi-huansu-H3', 52, 'SUV', 2),
    CarInfo('benchi-GLK-300', 53, 'SUV', 2),
    CarInfo('benchi-ML500', 54, 'SUV', 2),
    CarInfo('fengtian-puladuo-06', 55, 'SUV', 2),
    CarInfo('fengtian-SUV-gai', 56, 'SUV', 2),
    CarInfo('guangqi-chuanqi-GS4-2015', 57, 'SUV', 2),
    CarInfo('jianghuai-ruifeng-S3', 58, 'SUV', 2),
    CarInfo('jili-boyue', 59, 'SUV', 2),
    CarInfo('jipu-3', 60, 'SUV', 2),
    CarInfo('linken-SUV', 61, 'SUV', 2),
    CarInfo('lufeng-X8', 62, 'SUV', 2),
    CarInfo('qirui-ruihu', 63, 'SUV', 2),
    CarInfo('rongwei-RX5', 64, 'SUV', 2),
    CarInfo('sanling-oulande', 65, 'SUV', 2),
    CarInfo('sikeda-SUV', 66, 'SUV', 2),
    CarInfo('Skoda_Fabia-2011', 67, 'SUV', 2),
    CarInfo('xiandai-i25-2016', 68, 'SUV', 2),
    CarInfo('yingfeinidi-qx80', 69, 'SUV', 2),
    CarInfo('yingfeinidi-SUV', 70, 'SUV', 2),
    CarInfo('benchi-SUR', 71, 'SUV', 2),
    CarInfo('biyadi-tang', 72, 'SUV', 2),
    CarInfo('changan-CS35-2012', 73, 'SUV', 2),
    CarInfo('changan-cs5', 74, 'SUV', 2),
    CarInfo('changcheng-H6-2016', 75, 'SUV', 2),
    CarInfo('dazhong-SUV', 76, 'SUV', 2),
    CarInfo('dongfeng-fengguang-S560', 77, 'SUV', 2),
    CarInfo('dongfeng-fengxing-SX6', 78, 'SUV', 2)
]
NAME2ID = {info.name: info for info in MODELS}
ID2NAME = {info.id: info for info in MODELS}
