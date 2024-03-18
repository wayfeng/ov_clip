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
        cast_dtype = self.model.transformer.get_cast_dtype()
        x = self.model.token_embedding(text).to(
            cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.model.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.model.text_projection
        return x #F.normalize(x, dim=-1)


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
            t, example_input=token_input, input=(-1, 77))
        ov.save_model(openclip_text_encoder,
                      f"{args.output_path}/{model_id.lower().replace('-','_')}_text.xml")
    else:
        dummy_inputs = {
            "image": torch.randn(1, 3, 224, 224, dtype=torch.float32),
            "text": torch.randint(low=0, high=49407, size=(-1, 77)),
        }
        ov_model = ov.convert_model(
            model, example_input=dummy_inputs, input=([batch, 3, 224, 224], [10, 77]))
        ov.save_model(ov_model, f"{model_id.lower().replace('-','_')}.xml")
