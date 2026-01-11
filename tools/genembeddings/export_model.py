import torch
from sentence_transformers import SentenceTransformer

class SBERT(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.transformer = model[0].auto_model
        self.pooling = model[1]

    def forward(self, input_ids, attention_mask):
        out = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        sentence_emb = self.pooling({
            "token_embeddings": out.last_hidden_state,
            "attention_mask": attention_mask,
        })["sentence_embedding"]
        return sentence_emb

model = SentenceTransformer("all-MiniLM-L6-v2") 
model.eval()
tokens = model.tokenize(["hello world"])

torch.set_default_device("cpu")

torch.onnx.export(
    SBERT(model),
    (
        torch.tensor(tokens["input_ids"]),
        torch.tensor(tokens["attention_mask"]),
    ),
    "sbert.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["sentence_embedding"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "sentence_embedding": {0: "batch"},
    },
    opset_version=17,
)
