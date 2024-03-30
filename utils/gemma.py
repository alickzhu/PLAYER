from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_gemma_model(model_id):

    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        # torch_dtype=dtype,
    )
    return model, tokenizer

def gemma_predict(model, tokenizer, prompt, system_prompt, max_new_tokens):

    chat = [
        { "role": "user", "content": system_prompt + '\n\n' + prompt},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=max_new_tokens)
    reply = tokenizer.decode(outputs[0][inputs.size()[1]:], skip_special_tokens=True)
    return reply

