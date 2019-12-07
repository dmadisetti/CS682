import os
import numpy as np

PROVISIONING_SCRIPT = open(
    os.path.join(os.path.dirname(__file__), 'provision.sh')).read()

CAMERA = np.array(
    [[2304.5479, 0, 1686.2379], [0, 2305.8757, 1354.9849], [0, 0, 1]],
    dtype=np.float32)
