import faiss
from pathlib import Path
from tqdm import tqdm
from utils import load_model, preprocess_image, dump_pkl

def build_index(filelist, model):
    _, d = model.outputs[0].shape
    index = faiss.IndexFlatIP(d)
    ireq = model.create_infer_request()
    for image_file in tqdm(filelist):
        image = preprocess_image(image_file)
        embedding = ireq.infer({'x': image[None]}).to_tuple()[0]
        index.add(embedding)
    return index

if __name__ == '__main__':
    device = 'GPU.1' # or use 'CPU'
    if not Path('data').exists():
        Path('data').mkdir()
    filelist = [str(f) for f in Path('./images').rglob("*.jpg")]
    dump_pkl('data/files.pkl', filelist)
    model = load_model('models/clip_vit_h_14_visual.xml', device, throughputmode=True)
    index = build_index(filelist, model)
    dump_pkl('data/embeddings.pkl', index)