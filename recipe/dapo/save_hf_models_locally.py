from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Download into current directory
model_name = "Qwen/Qwen3-4B-Base"
local_dir = "./Qwen3-4B-Base"   # current directory

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=local_dir)


tokenizer.save_pretrained(local_dir)
model.save_pretrained(local_dir)
