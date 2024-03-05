# OpenCLIP with OpenVINO

Original OpenCLIP [repo](https://github.com/mlfoundations/open_clip.git)

## To convert model
```bash
python convert_clip.py ViT-B-32 -c <path/to/ckpt/file.pth> -o models/
```

## To run the model
```bash
python clip.py -v models/vit_b_32_visual.xml -t models/vit_b_32_text.xml -i <path/to/images/cat_dog.jpeg> -p "eagle,cat,tiger,dinosaurs" -d 'GPU'
```
