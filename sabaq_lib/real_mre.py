from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("camembert/camembert-base-wikipedia-4gb")
model = AutoModel.from_pretrained("camembert/camembert-base-wikipedia-4gb")

print("cuda" if torch.cuda.is_available() else "cpu")

input_ids = tokenizer("s", return_tensors="pt").to("cuda")

output = model(input_ids)
