from datasets import load_dataset
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
from nltk.tokenize import sent_tokenize


ds = load_dataset(
    "openwebtext",
    split="train",
    streaming=True,
)

chunks = []
ids = []

next_id = 0

for row in ds:
    for c in sent_tokenize(row["text"]):
        if 40 <= len(c) <= 200:
            chunks.append(c)
            ids.append(next_id)
            next_id += 1

    if len(chunks) >= 200000:
        break

chunks = chunks[:200000]
ids = ids[:200000]

import numpy as np

with open("openwebtext.txt", "w") as f:
    for c in chunks:
        f.write(c.replace("\n", " ") + "\n")

print("Number of chunks:", len(chunks))
print("Number of IDS:", len(ids))

np.array(ids, dtype="uint64").tofile("openwebtext-ids.bin")

from sentence_transformers import SentenceTransformer

device = "mps"  
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

embeddings = model.encode(
    chunks,
    batch_size=128,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

print("Number of embeddings:", embeddings.shape[0])
print("Embedding dim:", embeddings.shape[1])

embeddings = embeddings.astype("float32")
print("MB:", embeddings.nbytes / (1024 * 1024))
embeddings.tofile("openwebtext-embeddings.bin")
print("Saved embeddings to openwebtext-embeddings.bin")
