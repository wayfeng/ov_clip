import sys
import argparse
import torch
import openvino as ov
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
import warnings

warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

class Bert(torch.nn.Module):
    def __init__(self, model):#, tokenizer):
        super().__init__()
        self.model = model
    def forward(self, text):
        pad_index = self.model.tokenizer.vocab['[PAD]']
        attn_mask = text.ne(pad_index).type(self.model.dtype)
        x = self.model.bert(text, attention_mask=attn_mask)[0].type(self.model.dtype)
        return x[:, 0, :] @ self.model.text_projection

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='convert Chinese CLIP model to OpenVINO IR')
    parser.add_argument('model_id', default='ViT-H-14',
                        help='the CLIP model ID')
    parser.add_argument('-o', '--output_path',
                        default='./models', help='path to store output models')
    parser.add_argument('-b', '--batch', default=1, help='default batch size')
    parser.add_argument('--fp16', default=True,  action='store_true',
                        help='whether save model as fp16 mode')

    args = parser.parse_args()

    model_id = args.model_id
    if model_id not in available_models():
        parser.print_help()
        exit(0)

    model, preprocess = load_from_name(model_id, device='cpu', download_root='./checkpoints/')
    _ = model.eval()

    batch = args.batch
    # convert visual transformer
    image_input = {"x":torch.randn(1,3,224,224,dtype=model.dtype)}
    clip_image_encoder = ov.convert_model(model.visual, example_input=image_input, input=(1,3,224,224))
    ov.save_model(clip_image_encoder, f"{args.output_path}/cn_clip_{model_id.lower().replace('-','_')}_visual.xml")
    # convert text transformer
    token_input = {"text": torch.randint(low=672, high=21128, size=(1,52))}
    bert = Bert(model)
    clip_text_encoder = ov.convert_model(bert, example_input=token_input, input=(1, 52))
    ov.save_model(clip_text_encoder, f"{args.output_path}/cn_clip_{model_id.lower().replace('-','_')}_text.xml")

