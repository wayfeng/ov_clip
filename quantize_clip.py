import argparse
import openvino as ov
import logging
import cv2
import open_clip
import nncf
import numpy as np
import torch
import torch.utils.data as data
from torchvision.transforms.functional import to_pil_image
from zipfile import ZipFile
from pathlib import Path


def transform_fn(image_data):
    """
    Quantization transform function. Extracts and preprocess input data from dataloader item for quantization.
    Parameters:
        image_data: image data produced by DataLoader during iteration
    Returns:
        input_tensor: input data in Dict format for model quantization
    """
    return preprocess_image(to_pil_image(np.squeeze(image_data.numpy()))).unsqueeze(0)


class COCOLoader(data.Dataset):
    def __init__(self, images_path):
        self.images = list(Path(images_path).iterdir())

    def __getitem__(self, index):
        image_path = self.images[index]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __len__(self):
        return len(self.images)


def load_data(data_dir: Path):
    # DATA_URL = "https://ultralytics.com/assets/coco128.zip"
    # DATA_DIR = Path('./data')
    zipfile = data_dir/'coco128.zip'

    if not (data_dir / "coco128/images/train2017").exists():
        with ZipFile(zipfile, "r") as zip_ref:
            zip_ref.extractall(zipfile)
    else:
        print("File existed")

    coco_dataset = COCOLoader(data_dir / 'coco128/images/train2017')
    calibration_loader = torch.utils.data.DataLoader(coco_dataset)

    return nncf.Dataset(calibration_loader, transform_fn)


# Model tags
# "ViT-B-32": "laion2b_s34b_b79k"
# "ViT-L-14": "laion2b_s32b_b82k"
clip_models = {
    "vit-b-32": "./checkpoints/open_clip_vit_b_32.pth",
    "vit-l-14": "./checkpoints/open_clip_vit_l_14.pth",
    "vit-h-14": "./checkpoints/open_clip_vit_h_14.pth",
    "vit-g-14": "./checkpoints/open_clip_vit_g_14.pth",
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_id', default='ViT-B-32',
                        help="Model id to convert")
    parser.add_argument('-o', '--model_dir', default='/tmp',
                        help="Folder for OpenVINO IR")
    parser.add_argument('-d', '--data_dir', default='./data',
                        help="Data folder to calibrate model")
    args = parser.parse_args()

    model_id = args.model_id
    model, _, preprocess_image = open_clip.create_model_and_transforms(
        model_id, pretrained=clip_models[model_id.lower()])
    # tokenizer = open_clip.get_tokenizer(model_id)

    core = ov.Core()

    nncf.set_log_level(logging.ERROR)
    fp16_model_path = Path(args.model_dir) / \
        f"{model_id.lower().replace('-','_')}_visual.xml"
    int8_model_path = Path(args.model_dir) / \
        f"{model_id.lower().replace('-','_')}_visual_int8.xml"
    # calibration_data = prepare_dataset()
    ov_model = core.read_model(fp16_model_path)
    calibration_dataset = load_data(Path(args.data_dir))
    quantized_model = nncf.quantize(
        model=ov_model,
        calibration_dataset=calibration_dataset,
        model_type=nncf.ModelType.TRANSFORMER,
    )
    ov.save_model(quantized_model, int8_model_path)
