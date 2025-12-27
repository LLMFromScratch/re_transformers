import torch
from transformers import AutoTokenizer

from re_transformers import AlbertForMaskedLM


tokenizer = AutoTokenizer.from_pretrained("albert/albert-base-v2")
model = AlbertForMaskedLM.from_pretrained("albert/albert-base-v2")

inputs = tokenizer("The capital of [MASK] is Paris.", return_tensor="pt")
logits: torch.Tensor = model(**inputs).logits

mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(
    as_tuple=True
)[0]
predicted_token_id = logits[0, mask_token_index].argmax(dim=-1)
predicted_token = tokenizer.decode(predicted_token_id)
print(f"predicted_token = {predicted_token}")
