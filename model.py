from transformers import pipeline

print("Loading local model...")
# This downloads a lightweight model locally that runs completely offline/free on your machine
chatbot = pipeline("text-generation", model="sshleifer/tiny-gpt2")

prompt = "Hello! Can you tell me a quick joke?"
print(f"\nUser: {prompt}")

response = chatbot(prompt, max_new_tokens=50, do_sample=True)[0]['generated_text']
print(f"\nAI: {response}")
