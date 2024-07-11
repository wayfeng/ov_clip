from tokenizer import tokenize
from utils import load_clip_model
from pymilvus import MilvusClient
from pathlib import Path
from minio import Minio
from config import *
import argparse
import shutil
import gradio as gr
import numpy as np



def query_images(text, k=4):
    tokens = tokenize(text)
    text_features = tenc.infer_new_request(tokens)
    tfeat = text_features.to_tuple()[0]
    tfeat /= np.linalg.norm(tfeat, axis=1, keepdims=True)

    hits = milvus_client.search(
        collection_name, tfeat, limit=k, 
        search_param={
            "metric_type": "IP",
            "params": {}
        },
        output_fields=["uuid"])
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir()
    results = []
    for hit in hits:
        for h in hit:
            uid = h['entity']['uuid']
            tmp_file = tmp_path.joinpath(f"{uid}.jpg")
            minio_client.fget_object(collection_name, uid, tmp_file)
            results.append(tmp_file)

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser("OpenCLIP demo: WebUI", add_help=True)
    parser.add_argument('-c', '--collection', default='traffics', help='Milvus collection name')
    parser.add_argument('-d', '--device', default='CPU', help='Device to inference')
    parser.add_argument('-m', '--text_model_path', default='models/vit_h_14_text.xml', help='Path to text encoder model')
    parser.add_argument('-n', '--max_queries', default=20, help='Maximum number of queries')
    args = parser.parse_args()
    collection_name = args.collection
    txt_encoder_path = args.text_model_path
    device = args.device
    max_queries = int(args.max_queries)

    minio_client = Minio(MINIO_API_HOST, access_key="minioadmin", secret_key="minioadmin", secure=False)
    milvus_client = MilvusClient(uri=MILVUS_HOST, token="root:Milvus")
    _, tenc = load_clip_model(None, txt_encoder_path, device)

    has = milvus_client.has_collection(collection_name)
    if not has:
        print("embeddings not found")
        exit(-1)
    milvus_client.load_collection(collection_name)
    tmp_path = Path('/tmp/clip_demo_images')
    with gr.Blocks() as demo:
        text = gr.Textbox(label="prompt", scale=1)
        k = gr.Slider(minimum=1, maximum=max_queries, value=max_queries, step=1, label="output number")
        btn = gr.Button("Query")
        gallery = gr.Gallery(label="results")
        btn.click(fn=query_images, inputs=[text, k], outputs=gallery)

        demo.launch(server_name='0.0.0.0', server_port=7580, share=False, debug=False)

    if tmp_path.exists():
        shutil.rmtree(tmp_path)
