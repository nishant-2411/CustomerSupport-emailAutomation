import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "checkpoints/shipcube_2Dlora_model"

tokenizer = None
model = None

def load_model():
    global tokenizer, model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base, LORA_PATH)
    model.eval()
    print("Model loaded!")

def reply(email):
    prompt = f"""You are a logistics support assistant.

Customer email:
\"\"\"{email}\"\"\"

Reply professionally."""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7
        )
    
    response = tokenizer.decode(out[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    load_model()
    
    email = "My order 45678 is delivered to a wrong address"
    response = reply(email)
    print(response)