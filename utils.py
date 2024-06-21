import numpy as np
import openvino as ov
import openvino.properties.hint as hints
import pickle
from PIL import Image
from pathlib import Path

def normalize(arr, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
    arr = arr.astype(np.float32)
    arr /= 255.0
    for i in range(3):
        arr[...,i] = (arr[...,i] - mean[i]) / std[i]
    return arr

def preprocess_image(image_path, shape=[224,224]):
    raw_image = Image.open(image_path).convert('RGB')
    img = raw_image.resize(shape, Image.Resampling.NEAREST)
    img = normalize(np.asarray(img))
    return img.transpose(2,0,1)

def load_model(model_path, device='CPU', throughputmode=False):
    if not model_path or not Path(model_path).exists():
        return None
    core = ov.Core()
    if throughputmode:
        core.set_property(device,
          {hints.performance_mode: hints.PerformanceMode.THROUGHPUT})
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model, device.upper())
    return compiled_model

def load_pkl(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
        return obj

def dump_pkl(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=4)