import os
import pandas as pd
import json
from glob import glob
from pathlib import Path
import numpy as np


def load_data(data_path, train_path="train.csv"):
    data = pd.read_csv(os.path.join(data_path, train_path))
    # This may be the ugliest list comprehension I have ever written.
    data = pd.DataFrame(
        sum([[[tag, int(row[0]),
               np.array(list(map(float, row[1:])))]
              for row in np.reshape(batch, (-1, 7))] for tag, batch in zip(
                  data.ImageId, data.PredictionString.str.split(" "))], []),
        columns=["ImageId", "CarId", "Prediction"])
    return data


def load_jsons(json_path="train/json"):
    jsons = {}
    for f in glob(f"{json_path}/*json"):
        name = Path(f).name.split(".")[0]
        with open(f) as json_file:
            jsons[name] = json.load(json_file)
    return jsons
