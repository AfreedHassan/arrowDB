from sentence_transformers import SentenceTransformer
from sys import argv

if len(argv) != 2:
    print("Usage: uv run python query.py <query>")
    exit(1)

query = argv[1]

model = SentenceTransformer("all-MiniLM-L6-v2")

qvec = model.encode(
    [query],
    normalize_embeddings=True
).astype("float32")

qvec.tofile("query.bin")


