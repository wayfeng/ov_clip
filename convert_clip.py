import sys
import argparse
import torch
import torch.nn.functional as F
import open_clip
import openvino as ov
import warnings
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)


class TextTransformer(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='convert OpenCLIP model to OpenVINO IR')
    parser.add_argument('model_id', default='ViT-B-32',
                        help='the original OpenCLIP model ID')
    parser.add_argument('-c', '--ckpt_path', help='path to checkpoints')
    parser.add_argument('-o', '--output_path',
                        default='/tmp/openclip', help='path to store output models')
    parser.add_argument('-b', '--batch', default=1, help='default batch size')
    parser.add_argument('-s', '--seperated', default=True, action='store_true',
                        help='whether sperate the OpenCLIP model to image encoder and text encoder')
    parser.add_argument('--static_text', default=True,  action='store_true',
                        help='whether convert text encoder with static input size')
    parser.add_argument('--fp16', default=True,  action='store_true',
                        help='whether save model as fp16 mode')

    args = parser.parse_args()

    model_id = args.model_id
    if model_id not in ['ViT-B-32', 'ViT-L-14', 'ViT-H-14', 'ViT-g-14']:
        parser.print_help()
        exit(0)
    # print(f"{model_id}, {seperated}, {args.fp16}")
    pretrained = args.ckpt_path
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_id, pretrained=pretrained)
    batch = args.batch
    text_batch = -1
    if args.static_text:
        text_batch = batch
    if args.seperated:
        # convert visual transformer
        image_input = {"x": torch.randn(
            batch, 3, 224, 224, dtype=torch.float32)}
        openclip_image_encoder = ov.convert_model(
            model.visual, example_input=image_input, input=(batch, 3, 224, 224))
        ov.save_model(openclip_image_encoder,
                      f"{args.output_path}/{model_id.lower().replace('-','_')}_visual.xml")
        # convert text transformer
        t = TextTransformer(model)
        token_input = {"text": torch.randint(low=0, high=49407, size=(1, 77))}
        # openclip_text_encoder = ov.convert_model(t, example_input=token_input, input=(10,77))
        openclip_text_encoder = ov.convert_model(
            t, example_input=token_input, input=(text_batch, 77))
        ov.save_model(openclip_text_encoder,
                      f"{args.output_path}/{model_id.lower().replace('-','_')}_text.xml")
    else:
        dummy_inputs = {
            "image": torch.randn(1, 3, 224, 224, dtype=torch.float32),
            "text": torch.randint(low=0, high=49407, size=(text_batch, 77)),
        }
        ov_model = ov.convert_model(
            model, example_input=dummy_inputs, input=([batch, 3, 224, 224], [10, 77]))
        ov.save_model(ov_model, f"{model_id.lower().replace('-','_')}.xml")
