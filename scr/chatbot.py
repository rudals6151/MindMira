from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
text = "I'm depressed"
encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
output = model.generate(**encoded_input, max_length=50, num_return_sequences=1, do_sample=True)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)