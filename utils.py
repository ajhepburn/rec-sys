from pathlib import Path
import numpy as np
import json

def check_file(path, filename):
    my_file = Path(path+filename)
    if my_file.is_file(): return 1
    else: return 0

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)