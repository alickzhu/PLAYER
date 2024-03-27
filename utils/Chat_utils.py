import openai
import time
from utils.llama import load_llama_model, llama_predict
from utils.gemma import load_gemma_model, gemma_predict
from transformers import AutoTokenizer
import tiktoken


class ChatSession:

    def __init__(self, args, system='', model=None, tokenizer=None):
        self.is_english = args.is_english
        self.model, self.tokenizer = model, tokenizer

        self.model_type = args.model_type
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
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
            openai.api_version = ""
            openai.api_base = ''
            openai.api_key = ''
            self.engine = 'gpt3516k'

        elif self.model_type=='gpt4':
            openai.api_type = "azure"
            openai.api_version = ""
            openai.api_base = ''
            openai.api_key = ''
            self.engine = ''

    def get_reply(self, prompt, evaluate=False):
        if self.is_english:
            prompt = prompt.replace("Chinese", "English").replace("chinese", "English")

            self.system = self.system.replace("Always response in Chinese, DO NOT ANY Translation.", "")
        if 'gpt' in self.model_type:
            return self.get_reply_gpt(prompt,evaluate)
        elif 'gemma' in self.model_type:
            return self.get_reply_gemma(prompt,evaluate)
        elif 'llama' in self.model_type:
            return self.get_reply_llama(prompt,evaluate)

    def get_reply_llama(self, prompt, evaluate=False):
        system_len = len(self.tokenizer.encode(self.system))
        prompt_len = len(self.tokenizer.encode(prompt))
        max_new_tokens = 300
        predict = llama_predict(self.model, self.tokenizer, prompt, self.system, max_new_tokens)
        return predict

    def get_reply_gemma(self, prompt, evaluate=False):
        system_len = len(self.tokenizer.encode(self.system))
        prompt_len = len(self.tokenizer.encode(prompt))
        max_new_tokens = 300
        predict = gemma_predict(self.model, self.tokenizer, prompt, self.system, max_new_tokens)
        return predict

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



if __name__ == '__main__':
    class args:
        model_type = 'gpt35'
        ckpt_dir = '/scratch/prj/inf_llmcache/hf_cache/models--meta-llama--Llama-2-13b-chat-hf/snapshots/c2f3ec81aac798ae26dcc57799a994dfbf521496'

    system_prompt = '''  You are playing a game called "Murder Mystery" with other players, which is based on textual interaction. Here are the game rules:
    Rule 1: The total number of players participating in the game depends on the script. There may be one or more players who are the murderer(s), while the rest are civilians.
    Rule 2: The goal of the game is for civilian players to collaborate and face a meticulously planned murder case together, collecting evidence and reasoning to identify the real murderer among the suspects; murderer players must concoct lies to hide their identity and avoid detection, while also achieving other objectives in the game.
    Rule 3: Throughout the game, only murderer players are allowed to lie. To conceal their identity, murderers may choose to frame others to absolve themselves of guilt; non-murderer players (civilians) must answer questions from other players and the host honestly and provide as much information as they know about the case to help uncover the truth.
    Rule 4: The game host is only responsible for ensuring the game follows a specific process. They are not players in the game and do not participate in the storyline.
    Rule 5: At the start of the game, each player receives their personal character script from the host, which contains information about their role and identity.
    Rule 6: The game may have multiple acts, and your script will be updated accordingly.
    Rule 7: Other players cannot see the content of each player's personal character script, so players must and can only collect information about other players through interaction after the game starts.
    Rule 8: In the voting phase, each player needs to cast their vote for who they think is the murderer in each case (including themselves, although this is not encouraged). If the player with the most votes is the murderer, the civilian players win. Otherwise, the murderer players win.
  Gameplay:
    The game has one or more acts. At the beginning of the game, players introduce themselves according to the script, and in each act, you will receive more plot information. In each act, you can ask questions, share your observations, or make deductions to help solve the murder case.
    The goal is to identify the true murderer and explain their motive. If you are the true murderer, you must hide your identity and avoid detection.
  
  Now, you are playing the role of {character_name}, and the other players are {character_name_list}. 
  You are not the murderer. Please collaborate with the other civilian players to achieve your personal objective while finding the true culprit!
  Do not pretend you are other players or the moderator. Always response in Chinese.
'''
    prompt = '''  Your Script is 你姓“孙”，小名“甜儿”，生于清光绪十二年（1886年），是巢湖当地人，家住“荒山”北，父母去世得早，哥哥【孙咸恩】比你大10岁，去湖中画舫上做水夫赚钱养你，有时会把船上的好吃的装在一个食盒里，带回家给你。
你8岁时（1893年），哥哥为了娶妻，到“崔庄”借钱，欠下一笔债·······哥哥婚后不久，“崔庄”的护院【周蒙当】到你家讨利息，见你哥还不出，就找来一位“张婆子”，把你带走，送到债主的家里干活，充抵利息。
“张婆子”说你家根本还不起钱，不如把你卖的画舫上，多少能换回几两银子······多亏太太（汪氏）的丫鬟【宝柳】为你说话，才把你留下--太太根据你的小名，给你改名【宝柑】，做了丫鬟--你那时靠“宝柳姐”照顾，她教你如何小心做事，又安慰你说等你哥凑到钱，就能来接你······
次年（1894年），太太的妹妹【昔颜】被接来“崔庄”住，老爷【崔寿亨】让你去贴身服侍【昔颜】，和她一起住“北院”（昔颜房）。.

  If you have something to hide, then be sure not to divulge the relevant information!
  
  Please introduce yourself according to your current script, introduce yourself as much as possible, and be careful to play your role according to your performance and your goal.
  Do not ask other people question at this stage.
'''

    a = ChatSession(args, system_prompt)
    print(a.get_reply(prompt))
    print(a.input_token)
    print(a.output_token)