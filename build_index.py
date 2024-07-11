import argparse
import os
import numpy as np
from minio import Minio
from tqdm import tqdm
from PIL import Image
from utils import load_clip_model, preprocess_image
from uuid import uuid4
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

fmt = "\n=== {:30} ===\n"
MINIO_API_HOST = "localhost:9000"
MILVUS_HOST = "http://localhost:19530"
INDEX_COLUMN = 'embedding'
BATCH_SIZE = 1000
DIM = 1024

def build_embeddings(image_path, image_encoder_path, collection_name, device='CPU', drop_existed=True):
    ie, _ = load_clip_model(image_encoder_path, '', device, throughputmode=True)
    ireq = ie.create_infer_request()
    schema = CollectionSchema([
            FieldSchema(name="uuid", dtype=DataType.VARCHAR, is_primary=True, max_length=32),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=DIM)
        ], "traffics infomation schema")
    has = utility.has_collection(collection_name)
    if has and drop_existed:
        utility.drop_collection(collection_name)
    collection = Collection(collection_name, schema, consistency_level="Strong")

    minio_client = Minio(MINIO_API_HOST, access_key="minioadmin", secret_key="minioadmin", secure=False)
    if not minio_client.bucket_exists(collection_name):
        minio_client.make_bucket(collection_name)

    paths = []
    for root, _, files in os.walk(image_path):
        for file in files:
            filename = os.path.join(root, file)
            paths.append(filename)
    for filename in tqdm(paths):
        image = preprocess_image(Image.open(filename))
        embedding = ireq.infer({'x': image[None]}).to_tuple()[0]
        embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
        uid = uuid4().hex
        collection.insert([[uid], embedding])
        minio_client.fput_object(collection_name, uid, filename)
    collection.flush()
    collection.create_index(
        field_name="embeddings",
        index_params={
            "index_type": "IVF_FLAT",
            "metric_type": "IP",
            "params": {"nlist": DIM},
        })

def main(device, image_path, image_model_path, collection):
    try:
        print(fmt.format("Connecting to Milvus server"))
        connections.connect("default", host="localhost", port="19530", token="root:Milvus")
    except:
        print(fmt.format("Failed to connect to Milvus"))
        exit(-1)

    print(fmt.format("Create embeddings"))
    build_embeddings(image_path, image_model_path, collection, device)
    print(fmt.format("Disconnecting from Milvus"))
    connections.disconnect("default")
    print(fmt.format("Done"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser("OpenCLIP demo: build faiss index", add_help=True)
    parser.add_argument('-c', '--collection', default='traffics', help='Milvus collection name')
    parser.add_argument('-d', '--device', default='CPU', help='Device to inference')
    parser.add_argument('-i', '--image_path', default='images', help='Path to images')
    parser.add_argument('-m', '--image_model_path', default='models/vit_h_14_visual.xml', help='Path to image encoder model')
    args = parser.parse_args()

    main(args.device, args.image_path, args.image_model_path, args.collection)
