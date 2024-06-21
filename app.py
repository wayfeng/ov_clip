from tokenizer import tokenize
from utils import load_model, load_pkl
import gradio as gr
#import numpy as np

def query_images(text, k=4):
    tokens = tokenize(text)
    text_features = model.infer_new_request(tokens)
    tfeat = text_features.to_tuple()[0]
    #tfeat /= np.linalg.norm(tfeat, axis=1, keepdims=True)
    _, indices = index.search(tfeat, k)
    images = []
    for i in map(int, indices[0]):
        images.append(filelist[i])
    return images

if __name__ == '__main__':
    model = load_model('models/clip_vit_h_14_text.xml')
    filelist = load_pkl('./data/files.pkl')
    index = load_pkl('./data/embeddings.pkl')
    with gr.Blocks() as demo:
        gr.Markdown("# Multi-Modal Image Searching Demo")
        text = gr.Textbox(label="prompt")
        k = gr.Slider(minimum=1, maximum=10, value=2, step=1, label="output number")
        btn = gr.Button("Query")
        gallery = gr.Gallery(label="results", columns=2)
        btn.click(fn=query_images, inputs=[text, k], outputs=gallery)

        demo.launch(server_port=7580)
