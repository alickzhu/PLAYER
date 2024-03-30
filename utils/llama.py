# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch



from typing import List, Literal, TypedDict
from transformers import LlamaTokenizer
import torch
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
def format_tokens(dialogs, tokenizer):
    prompt_tokens = []
    for dialog in dialogs:
        if dialog[0]["role"] == "system":
            dialog = [
                         {
                             "role": dialog[1]["role"],
                             "content": B_SYS + dialog[0]["content"] + E_SYS  + dialog[1]["content"],
                         }
                     ] + dialog[2:]
        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system','user' and 'assistant' roles, "
            "starting with user and alternating (u/a/u/a/u...)"
        )
        """
        Please verify that your tokenizer support adding "[INST]", "[/INST]" to your inputs.
        Here, we are adding it manually.
        """
        dialog_tokens: List[int] = sum(
            [
                tokenizer.encode(
                    f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                ) + [tokenizer.eos_token_id]
                for prompt, answer in zip(dialog[::2], dialog[1::2])
            ],
            [],
        )
        assert (
                dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        dialog_tokens += tokenizer.encode(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
        )
        prompt_tokens.append(dialog_tokens)
    return prompt_tokens



def load_llama_model(model_name, quantization=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"pad_token": "<PAD>",})

    return model, tokenizer

def llama_predict(
        model,
        tokenizer,
        prompt,
        system_prompt,
        max_new_tokens = 1000, #The maximum numbers of tokens to generate
        seed: int=42, #seed value for reproducibility
        do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
        use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
        top_p: float=0.9, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        temperature: float=0.75, # [optional] The value used to modulate the next token probabilities.
        top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
        repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
        length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation.
        **kwargs
):


    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    dialogs = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
    ]
    chats = format_tokens(dialogs, tokenizer)

    with torch.no_grad():

        tokens= torch.tensor(chats[0]).long()
        tokens= tokens.unsqueeze(0)
        tokens= tokens.to("cuda:0")
        outputs = model.generate(
            input_ids=tokens,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            use_cache=use_cache,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            **kwargs
        )

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)


        # print(f"Model output:\n{output_text}")
        output = output_text.split('[/INST]')[-1]
        return output




if __name__ == "__main__":
    model_name = ''
    model = load_model(model_name)
    llama_predict(model)

