from qdrant_client import models, QdrantClient
import numpy as np
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

qdrant = QdrantClient(":memory:")

encoder = SentenceTransformer('all-MiniLM-L6-v2') 

def upload_embd_get_similarity(user_ans,gpt_ans):
    qdrant.recreate_collection(
	collection_name="m",
	vectors_config=models.VectorParams(
		size=encoder.get_sentence_embedding_dimension(),
		distance=models.Distance.COSINE))

    qdrant.upload_records(
	collection_name="m",
	records=[
		models.Record(
			id=1,
			vector=encoder.encode(user_ans).tolist(),
			payload={'text':user_ans})])

    hits = qdrant.search(
	collection_name="m",
	query_vector=encoder.encode(gpt_ans).tolist(),
	limit=1)

    for hit in hits:
        return hit.score


print('ENDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD')
