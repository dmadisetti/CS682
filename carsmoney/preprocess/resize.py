import numpy as np
import cv2
import dask.dataframe as dd
from carsmoney.preprocess.utils import load_data


def create_resized(data_path="train/data",
                 train_path="train.csv",
                 width=384,
                 height=256,
                 partitions=30):
    """Generates Segmenation masks from dataset provided a path."""
    train = load_data(data_path, train_path)

    def process(batch):
        image_id = batch["ImageId"].iloc[0]
        image = cv2.imread(f"{data_path}/{image_id}.jpg",
                           cv2.COLOR_BGR2RGB)[:, :, ::-1]
        image = cv2.resize(image, (width, height))
        cv2.imwrite(f"{data_path}/{image_id}_{width}x{height}.jpg", image)
        return True

    dd.from_pandas(
        train, npartitions=partitions).groupby("ImageId").apply(
            process, meta=('processes', bool)).compute()
