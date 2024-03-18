import argparse

from PIL import Image
from scipy.special import softmax
from tokenizer import tokenize

import openvino as ov
import numpy as np
import openvino.properties.hint as hints

def load_model(image_encoder_path, text_encoder_path, device, throughputmode=False):
    core = ov.Core()
    #core.set_property(device, {hints.inference_precision: ov.Type.f32})
    ie = core.read_model(image_encoder_path)
    te = core.read_model(text_encoder_path)
    config = {}
    if throughputmode:
        config["PERFORMANCE_HINT"] = "THROUGHPUT"
    ienc = core.compile_model(ie, device.upper(), config)
    tenc = core.compile_model(te, device.upper(), config)
    #model.max_text_len = 256
    return ienc, tenc

def normalize(arr, mean=(0,0,0), std=(1,1,1)):
    arr = arr.astype(np.float32)
    arr /= 255.0
    for i in range(3):
        arr[...,i] = (arr[...,i] - mean[i]) / std[i]
    return arr

def preprocess_image(input_image, shape=[224,224]):
    img = input_image.resize(shape, Image.Resampling.NEAREST)
    img = np.asarray(img)
    img = normalize(img, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    return img.transpose(2,0,1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run OpenCLIP with OpenVINO')
    parser.add_argument('-v', '--visual_model_path', help='Path to OpenCLIP visual encoder models')
    parser.add_argument('-t', '--text_model_path', help='Path to OpenCLIP text encoder model')
    parser.add_argument('-i', '--image_path', help='Path to image')
    parser.add_argument('-d', '--device', help='Select device to execute')
    parser.add_argument('-p', '--prompt', help='Text prompt')

    args = parser.parse_args()

    image = preprocess_image(Image.open(args.image_path))
    tokens = tokenize(args.prompt.split(','))

    print("Loading Model...")
    ienc, tenc = load_model(args.visual_model_path, args.text_model_path, args.device)

    print("Inferencing...")
    image_feature = ienc.infer_new_request({"x": image[None]})
    text_feature = tenc.infer_new_request(tokens)

    tfeat = text_feature.to_tuple()[0]
    ifeat = image_feature.to_tuple()[0]
    probs = softmax(100.0 * ifeat @ tfeat.T)

    print([x for x in probs[0]])
