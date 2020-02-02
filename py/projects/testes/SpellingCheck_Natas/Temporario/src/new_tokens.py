import torch
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertModel.from_pretrained("bert-base-cased")

print(len(tokenizer))  # 28996
tokenizer.add_tokens(["NEW_TOKEN"], add_prefix_space=True)
print(len(tokenizer))  # 28997

model.resize_token_embeddings(len(tokenizer))
# The new vector is added at the end of the embedding matrix

print(model.embeddings.word_embeddings.weight[-1, :])
# Randomly generated matrix

model.embeddings.word_embeddings.weight[-1, :] = torch.zeros([model.config.hidden_size])

print(model.embeddings.word_embeddings.weight[-1, :])
# outputs a vector of zeros of shape [768]