from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("../weights/tokenizer")

text = "A cat is on the mat."

tokens = tokenizer.tokenize(text)
ids = tokenizer.encode(text)

print("Tokens:", tokens)
print("IDs:", ids)