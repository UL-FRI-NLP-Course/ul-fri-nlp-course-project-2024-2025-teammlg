import transformers
import torch
import time
import threading

start_time = time.perf_counter()

torch.manual_seed(30)
print("Transformers version: ", transformers.__version__)

model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

print("Loading tokenizer....")

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
#quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

print("Loading model....")
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
    #quantization_config=quantization_config
)
print("Model loaded!")

chat = [
    {"role": "user", "content": "Hi, how are you?"},
]

print("Setting inputs....")

text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

streamer = transformers.TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

inputs = tokenizer([text], return_tensors="pt").to('cuda')

generation_arguments = {
    'max_new_tokens': 512,
    'streamer': streamer,
    **inputs
}

thread = threading.Thread(
    target=model.generate,
    kwargs=generation_arguments
)
thread.start()

for text_token in streamer:
    time.sleep(0.01)
    print(text_token, end="")
print()

thread.join()

end_time = time.perf_counter()

print(f"Operations took {end_time - start_time} s")