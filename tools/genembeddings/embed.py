from datasets import load_dataset
import numpy as np


ds = load_dataset(
    "sciq",
    split="train",
    streaming=True,
)

ds = ds.filter(lambda x: x["support"] != "")
for i, q in enumerate(ds['question']):
    print(q)
    if i > 10:
        break

'''
chunks = []
ids = []

for (i, row) in enumerate(ds):
    chunks.append(row["support"])
    ids.append(i+1)



with open("sciq.txt", "w") as f:
    for c in chunks:
        f.write(c.replace("\n", " ") + "\n")

print("Number of chunks:", len(chunks))
print("Number of IDS:", len(ids))

np.array(ids, dtype="uint64").tofile("sciq-ids.bin")

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
embeddings.tofile("sciq-embeddings.bin")
print("Saved embeddings to openwebtext-embeddings.bin")
'''
