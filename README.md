# OpenCLIP with OpenVINO

Original OpenCLIP [repo](https://github.com/mlfoundations/open_clip.git) and [models](https://huggingface.co/models?library=open_clip&sort=trending&search=laion2b).

## To convert model
```bash
pip install -e git+https://github.com/mlfoundations/open_clip.git#egg=open_clip_torch
python convert_clip.py ViT-B-32 -c <path/to/ckpt/file.pth> -o models/
```

## To run the model
```bash
python clip.py -v models/vit_b_32_visual.xml -t models/vit_b_32_text.xml -i <path/to/images/cat_dog.jpeg> -p "eagle,cat,tiger,dinosaurs" -d 'GPU'
```

## Run benchmark
```bash
benchmark_app -m models/vit_l_14_visual.xml -hint latency -d GPU -data_shape "x[1,3,224,224]"
benchmark_app -m models/vit_l_14_text.xml -hint latency -d GPU -data_shape "text[1,77]"
```

