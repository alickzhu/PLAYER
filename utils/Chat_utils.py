import openai
import time

from openai import OpenAI
import random
import time
model_dict = {
    "qwen2.5_32B_0":{'model':'Qwen','base_url':"http://localhost:9999/v1"},
    "qwen2.5_32B_1":{'model':'Qwen2','base_url':"http://localhost:9998/v1"},
}
def select_random_url():
    url1 = "http://35.177.88.172:9001/v1"
    url2 = "http://35.177.88.172:9000/v1"
    current_second = time.time() % 60  # 获取当前时间的秒数
    return url1 if int(current_second) % 2 == 0 else url2

class ChatSession:

    def __init__(self, args, system='', model=None, tokenizer=None):
        self.args = args
        self.model_type = args.model_type
        self.is_english = args.is_english
        self.model, self.tokenizer = model, tokenizer

        self.try_time = 5
        if system == '':
            self.system = "Assistant is a large language model trained by OpenAI."
        else:
            self.system = system
        self.input_token = []
        self.output_token = []
        self.input_token_eva = []
        self.output_token_eva = []

    def set_gpt_version(self):
        if self.model_type=='gpt35':
            openai.api_type = "azure"
            openai.api_version = "2023-05-15"
            openai.api_base = ''
            openai.api_key = ''
            self.engine = 'gpt3516k'

        elif self.model_type=='gpt4':
            openai.api_type = "azure"
            openai.api_version = "2023-05-15"
            openai.api_base = ''
            openai.api_key = ''
            self.engine = ''

    def get_reply(self, prompt, evaluate=False, return_prob=False):
        if self.is_english:
            prompt = prompt.replace("Chinese", "English").replace("chinese", "English")

            self.system = self.system.replace("Always response in Chinese, DO NOT ANY Translation.", "")
        if 'gpt' in self.model_type:
            return self.get_reply_gpt(prompt,evaluate)

        elif "vllm" in self.model_type:
            return self.vllm_openai_api(prompt,evaluate,return_prob)

    def get_reply_gpt(self, prompt, evaluate=False):
        self.set_gpt_version()
        if evaluate:
            self.input_token_eva.append(len(self.tokenizer.encode(prompt)))
        else:
            self.input_token.append(len(self.tokenizer.encode(prompt)))
        count = 0
        while count < self.try_time:
            reply = ''
            try:
                response = openai.ChatCompletion.create(
                    engine=self.engine,

                    messages=[
                        {"role": "system",
                         "content": self.system},
                        {"role": "user", "content": prompt}
                    ]
                )
                reply = response.choices[0].message.content
            except:
                time.sleep(10)
            count += 1
            if reply != '':
                break
        if evaluate:
            self.output_token_eva.append(len(self.tokenizer.encode(reply)))
        else:
            self.output_token.append(len(self.tokenizer.encode(reply)))
        return reply
    def vllm_openai_api(self, prompt, evaluate=False,return_prob=False):
        count = 0
        extra_params = {}
        if return_prob:
            extra_params = {
                "logprobs": True,
                "top_logprobs": 20,
            }
        while count < 5:
            reply = ''
            try:
                if evaluate:
                    n = self.args.eva_times
                else:
                    n = 1

                if "llama" == self.args.model_name:
                    client = OpenAI(
                        api_key="EMPTY",
                        base_url="http://localhost:8000/v1",
                    )


                    completion = client.chat.completions.create(
                        model="meta-llama/Llama-3.1-70B-Instruct",
                        messages=[
                            {"role": "system", "content": self.system},
                            {"role": "user", "content": prompt},
                        ],
                        n=1,
                        temperature=self.args.temperature,
                        **extra_params
                    )
                elif "qwen" == self.args.model_name:
                    client = OpenAI(
                        api_key="EMPTY",
                        base_url="http://localhost:9999/v1",
                    )

                    completion = client.chat.completions.create(
                        model="Qwen",
                        messages=[
                            {"role": "system", "content": self.system},
                            {"role": "user", "content": prompt},
                        ],
                        n=n,
                        temperature=self.args.temperature,
                        **extra_params
                    )
                elif "qwen2" == self.args.model_name:
                    client = OpenAI(
                        api_key="EMPTY",
                        base_url="http://localhost:9998/v1",
                    )

                    completion = client.chat.completions.create(
                        model="Qwen2",
                        messages=[
                            {"role": "system", "content": self.system},
                            {"role": "user", "content": prompt},
                        ],
                        n=n,
                        temperature=self.args.temperature,
                        **extra_params
                    )
                elif "qwen3" == self.args.model_name:
                    client = OpenAI(
                        api_key="EMPTY",
                        base_url="http://localhost:9997/v1",
                    )

                    completion = client.chat.completions.create(
                        model="Qwen3",
                        messages=[
                            {"role": "system", "content": self.system},
                            {"role": "user", "content": prompt},
                        ],
                        n=n,
                        temperature=self.args.temperature,
                        **extra_params
                    )


                choices = completion.choices
                if len(choices) == 1:
                    reply = completion.choices[0].message.content
                else:
                    reply = [choice.message.content for choice in choices]

            except:
                print("Connection Error, Wait for 3 seconds")
                time.sleep(random.randint(1, 5))
            count += 1
            if reply != '':
                break
        if reply == '':
            reply = "Sorry Connection Error, Please Try Again Later"
        if return_prob:
            return reply, completion.choices[0].logprobs.content[0].top_logprobs
        return reply


if __name__ == '__main__':
