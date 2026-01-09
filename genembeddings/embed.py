from datasets import load_dataset


ds = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train",
    streaming=True,
)

texts = []
ids = []

next_id = 0

for row in ds:
    for sent in row["text"].split(". "):
        if len(sent) > 40:
            texts.append(sent)
            ids.append(next_id)
            next_id += 1

        if len(texts) >= 65000:
            break
    if len(texts) >= 65000:
        break


import numpy as np

with open("wikitext.txt", "w") as f:
    for t in texts:
        f.write(t.replace("\n", " ") + "\n")

print("Collected {} sentences".format(len(texts)))

np.array(ids, dtype="uint64").tofile("ids.bin")

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
)

embeddings = embeddings.astype("float32")
print("MB:", embeddings.nbytes / (1024 * 1024))
embeddings.tofile("embeddings.bin")
print("Saved embeddings to embeddings.bin")
