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


# from huggingface_hub import snapshot_download
#
# repo_id = "google/gemma-7b-it"  # 模型在huggingface上的名称
# local_dir = "/scratch/prj/lmrep/llama2_model/gemma-7b-it"  # 本地模型存储的地址
# local_dir_use_symlinks = False  # 本地模型使用文件保存，而非blob形式保存
# token = "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"  # 在hugging face上生成的 access token
#
# # # 如果需要代理的话
# # proxies = {
# #     'http': 'XXXX',
# #     'https': 'XXXX',
# # }
#
# snapshot_download(
#     cache_dir="/scratch/prj/lmrep/llama2_model/cache",
#     repo_id=repo_id,
#     local_dir=local_dir,
#     local_dir_use_symlinks=local_dir_use_symlinks,
#     token=token,
#     # proxies=proxies
# )
