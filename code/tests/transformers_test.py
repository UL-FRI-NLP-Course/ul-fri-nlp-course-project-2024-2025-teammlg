import transformers
import torch

torch.manual_seed(30)
print(transformers.__version__)


model_id = "deepseek-ai/DeepSeek-R1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model = model.to(device)

chat = [
    {
        "role": "user",
        "content": "Hi, how are you?"
    }
]
question = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
question = tokenizer(question, return_tensors="pt").to(device)

streamer = transformers.TextIteratorStreamer(tokenizer, skip_prompt=True)

for new_text in streamer:
    print(new_text, end="")
print()