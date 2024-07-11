import numpy as np
import openvino as ov
import openvino.properties.hint as hints
from PIL import Image
from pathlib import Path

def normalize(arr, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
    arr = arr.astype(np.float32)
    arr /= 255.0
    for i in range(3):
        arr[...,i] = (arr[...,i] - mean[i]) / std[i]
    return arr

def preprocess_image(input_image, shape=[224,224]):
    img = input_image.resize(shape, Image.Resampling.NEAREST)
    img = normalize(np.asarray(img))
    return img.transpose(2,0,1)


def load_clip_model(img_encoder_path, txt_encoder_path, device='GPU', throughputmode=False):
    core = ov.Core()
    # force inferencing in f32 mode for bug in f16 mode
    # core.set_property(device, {hints.inference_precision: ov.Type.f32})
    if throughputmode:
        core.set_property(device, {hints.performance_mode: hints.PerformanceMode.THROUGHPUT})
    ienc, tenc = None, None
    if img_encoder_path and Path(img_encoder_path).exists():
        ie = core.read_model(img_encoder_path)
        ienc = core.compile_model(ie, device.upper())
    if txt_encoder_path and Path(txt_encoder_path).exists():
        te = core.read_model(txt_encoder_path)
        tenc = core.compile_model(te, device.upper())
    return ienc, tenc
