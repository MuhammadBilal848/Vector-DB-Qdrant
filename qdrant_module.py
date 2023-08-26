from qdrant_client import models, QdrantClient
import numpy as np
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import ViTImageProcessor, ViTModel
import cv2
import numpy as np
import time
import os

qdrant = QdrantClient(":memory:")

def upload_qdrant(embedding_path,col_name,emb_size=768):
    ''' The function uploads embedding in the qdrant based on fixed set of parameters
    embedding_path : Path of registration car embedding of shape (n,768) in npy format
    col_name : Any name you want your collection to have     
    emb_size : Size of your embedding (default = 768)
     '''

    coll_emb = np.load(embedding_path)

    qdrant.recreate_collection(
	collection_name=col_name,
	vectors_config=models.VectorParams(
		size=emb_size,
		distance=models.Distance.COSINE
	))

    for idx, embd in enumerate(coll_emb):
      filename = f"{col_name}_{idx}.npy"
      payload = {"filename": filename}

      qdrant.upload_records(
          collection_name=col_name,
          records=[
              models.Record(
                  id=idx,
                  vector=embd.tolist(),
                  payload=payload
              )
          ]
      )

def do_similarity(target_folder,col_name):
    '''The function searches given embedding in the given collection 
    target_folder : Path of a folder having npy embedddings of shape (768,)
    col_name : Same collection that is set when uploading
    '''
    image_paths = []
    for root, _, filenames in os.walk(target_folder):
      for filename in filenames:
        if filename.lower().endswith('.npy'):
            image_path = os.path.join(root, filename)
            image_paths.append(image_path)

    d = []
    for ip in image_paths:
      test = np.load(ip)
      hits = qdrant.search(
          collection_name=col_name,
          query_vector=test,
          limit=1)
      
      for hit in hits:
        l = {'Filename':str(hit.payload['filename']+' Vs '+ip),
            'Score':hit.score}
        d.append(l)
    
    ret = pd.DataFrame(d)
    return ret