import argparse
import json
import os

from utils.json_tools import json_format
from tqdm import tqdm
from Game_Framework import Agent, ChatItem, EvaluateItem, Game, ScriptManager

class Game_think_thrice(Game):
    def __init__(self, scriptmanager, args):
        # 初始化父类
        super().__init__(scriptmanager, args)


    def self_introduction_phase(self):
        stage = 'self introduction stage'
        for i in tqdm(range(len(self.agents))):
            # Add host dialogue
            # self.add_chat(ChatItem(stage, "hold", self.agents[i].name, False,self.prompts['hold_intro_yourself'].format(character=self.agents[i].name)))

            if  not args.is_english:
                current_script = self.agents[i].database.query('你是谁？', 12000)
            else:
                current_script = self.agents[i].database.query('who are you?', 12000)

            if self.agents[i].is_murderer:
                intro_prompt = self.prompts['self_intro_intro'].format(current_script=current_script,   goal=self.agents[i].acts_goal)
            else:
                intro_prompt = self.prompts['self_intro_intro_murder'].format(current_script=current_script,   goal=self.agents[i].acts_goal)
            intro = self.agents[i].speak(intro_prompt)
            # intro_pro = self.process_reply(intro)
            self.add_chat(ChatItem(stage, self.agents[i].name, "all", False, intro,intro_prompt, is_ask=2))
            self.add_chat_database(1)

    def open_talk(self):
        for i in tqdm(range(len(self.agents))):
            for x,victim in enumerate(self.agents[i].victims):
                # get question
                for j in tqdm(range(len(self.agents)), desc='Character {0} investigate {1}'.format(self.agents[i].name, victim)):
                    stage = '{0} ask {1} for death of {2}'.format(self.agents[i].name, self.agents[j].name, victim)
                    if i == j:
                        continue
                    if self.agents[j].name not in self.agents[i].character_suspect[x]:
                        continue
                    # question_, question_prompt = self.propose_question_talk_stage(i)
                    current_script = self.agents[i].database.query(self.agents[j].name+','+victim, self.rsl)
                    chat_history = self.chat_dataset.query_num(self.agents[j].name+','+victim, 5)
                    if args.is_english:
                        prompt = self.prompts['Prompting_LLMs_to_select_questions'].replace('{current_script}',current_script).replace('{victim}',victim).replace('{chat_history}',chat_history).replace('{questions_prepared_for_specific_role}',str(defined_questions["English"]))
                    else:
                        prompt = self.prompts['Prompting_LLMs_to_select_questions'].replace('{current_script}',current_script).replace('{victim}',victim).replace('{chat_history}',chat_history).replace('{questions_prepared_for_specific_role}',str(defined_questions["Chinese"]))
                    questions = self.agents[i].speak(prompt)
                    questions = questions.split('#')
                    prompt2 = self.prompts['Prompting_LLMs_to_ask_questions'].replace('{current_script}',current_script).replace('{victim}',victim).replace('{chat_history}',chat_history).replace('{selected_questions}',str(questions))
                    question = self.agents[i].speak(prompt2)
                    question = self.process_reply(question)

                    self.add_chat(ChatItem(stage, self.agents[i].name, self.agents[j].name, False, question, prompt2, is_ask=1))
                    # get relay agent
                    relay_agent = self.agents[j]
                    chat_history = self.chat_dataset.query_num(question, 5)

                    if relay_agent.kill_by_me[x] == 1:
                        reply_prompt = self.prompts['open_talk_reply_murder'].format(current_script=self.agents[j].summary,goal=relay_agent.acts_goal,chat_history=chat_history,character=self.agents[i].name,question=question, victim=victim)
                        reply = relay_agent.speak(reply_prompt)
                    # reply:
                    else:
                        reply_prompt = self.prompts['Prompting_LLMs_to_generate_answers'].format(current_script=current_script,question=question,chat_history=chat_history,character =self.agents[i].name)
                        # reply_prompt = self.prompts['open_talk_reply_rap'].format(current_script=current_script,goal=relay_agent.acts_goal,chat_history=chat_history,character=self.agents[i].name,question=question)
                        reply = relay_agent.speak(reply_prompt)
                        try:
                            prompt = self.prompts['Prompting_LLMs_to_make_reflection'].replace('{chat_history}',chat_history)
                            reflection = self.agents[i].speak(prompt)
                            prompt = self.prompts['Prompting_LLMs_to_generate_the_final_response_'].replace('{reflection}',reflection).replace('{answer}',reply).replace('{character}',self.agents[i].name).replace('{question}',question)
                            reply = relay_agent.speak(prompt)
                            reply = json_format(reply)
                            reply = reply['my_final_answer']
                        except:
                            pass

                    self.add_chat(ChatItem(stage, relay_agent.name, self.agents[i].name, False, reply, reply_prompt, is_ask=0))
                    self.add_chat_database(2)

    def process_reply(self, reply):
        # Split the string based on the first occurrence of ":" or "："
        parts = reply.split(':')[-1] if ':' in reply else reply
        parts = parts.split('：')[-1] if '：' in parts else parts

        # Remove text within the last set of parentheses
        if '(' in parts and ')' in parts:
            start = parts.rfind('(')
            end = parts.rfind(')')
            parts = parts[:start] + parts[end+1:]
        elif '（' in parts and '）' in parts:
            start = parts.rfind('（')
            end = parts.rfind('）')
            parts = parts[:start] + parts[end+1:]

        return parts

    def start(self):
        self.self_introduction_phase()
        count = 0
        while count < self.args.max_turn:
            self.open_talk()
            count += 1

    def save_args_to_json(self, args, filename):
        args_dict = vars(args)
        with open(filename, 'w') as file:
            json.dump(args_dict, file, indent=4, ensure_ascii=False)

    def mkdir_output_dir(self, args, post = ''):
        args.output_root_path = os.path.join(args.output_root_path, args.script_name+post)

        if not os.path.exists(args.output_root_path):
            os.makedirs(args.output_root_path)

def evalute_para(game, rsl, rchl, path, prefix = ''):
    game.args.rsl = rsl
    game.args.rchl = rchl
    game.evaluate()
    game.save_evaluate_history(path, prefix)

def main():

    """play game"""
    scriptmanager = ScriptManager(args)
    game = Game_think_thrice(scriptmanager, args)
    game.mkdir_output_dir(args)
    game.start()

    game.save_history(args.output_root_path)

    path = args.output_root_path

    game.evalute_para(4000, 4000, path)
    # game.save_all_evaluate(path)
    game.save_params(args)

if __name__ == "__main__":

    defined_questions = {
        "English": [
            "What was your timeline on the day of the incident?",
            "How would you describe your relationship with the victim? Were there any conflicts or particularly close moments?",
            "When was the last time you saw the victim? What was their emotional and physical state then?",
            "Do you know if the victim had any enemies or conflicts with anyone?",
            "What details or anomalies did you notice at the scene of the crime?",
            "Did the victim mention anything to you or others that made them worried or fearful recently?",
            "Did you notice any unusual people or behaviors on the day of the incident?",
            "How much do you know about the victim's secrets or personal life?",
            "Were there any items or remains found at the crime scene that could be related to the crime?",
            "Do you have any personal opinions or theories about the case?"
        ],
        "Chinese": [
            "你在案发当天的时间线是怎样的？",
            "你与死者的关系如何？有没有发生过任何争执或特别亲近的时刻？",
            "你最后一次见到死者是什么时候？他们当时的情绪和状态怎样？",
            "你知道死者有没有敌人或与人有冲突吗？",
            "你发现了案发现场的哪些细节或异常之处？",
            "死者最近有没有跟你或其他人提及过令他们担心或恐惧的事情？",
            "你是否注意到案发当天有无不寻常的人物或行为出现？",
            "关于死者的秘密或个人生活，你了解多少？",
            "案发现场有没有发现任何可能与犯罪有关的物品或遗留物？",
            "你对这起案件有什么自己的看法或理论吗？"
        ]
    }

    parser = argparse.ArgumentParser()
    # for script
    parser.add_argument('--script_root', default=r"./chinese", type=str,required=False)
    parser.add_argument('--database_root', default=r"./database", type=str,required=False)
    parser.add_argument('--script_name', default="134-致命喷泉（4人封闭）", type=str, required=False)
    parser.add_argument('--prompt_path', default="./prompts_Werewolf.yaml", type=str,required=False)
    parser.add_argument('--sensor_path', default="./sensors.json", type=str,required=False)

    parser.add_argument('--load_history', default="134-致命喷泉（4人封闭）Werewolf", type=str,required=False)

    # for model
    parser.add_argument('--temperature', default=0.8, type=float, help='the diversity of generated text')
    parser.add_argument('--model_type', default="vllm", type=str, required=False, help='[gpt35, gpt4, llama13b]')
    parser.add_argument('--model_name', default="qwen2", type=str, required=False, help='[gpt35, gpt4, llama13b]')

    # for game
    parser.add_argument('--rsl', default=4000, type=float, help='the max length for relative script length (retrival from current script by RAG')
    parser.add_argument('--rchl', default=4000, type=float, help='the max length for relative chat history length (retrival from chat history by RAG')
    parser.add_argument('--each_rsl', default=3000, type=float, help='the max length for each agent\'s relative script length (retrival from current script by RAG')
    parser.add_argument('--question_number', default='1', type=str, help='the max length for each agent\'s relative script length (retrival from current script by RAG')
    parser.add_argument('--max_turn', default=3, type=int, help='the max length for each agent\'s relative script length (retrival from current script by RAG')
    parser.add_argument('--constraint', default=1, type=int, help='the max length for each agent\'s relative script length (retrival from current script by RAG')
    parser.add_argument('--sensor', default=1, type=int, help='the max length for each agent\'s relative script length (retrival from current script by RAG')
    parser.add_argument('--is_english', default=0, type=int, help='the max length for each agent\'s relative script length (retrival from current script by RAG')
    parser.add_argument('--eva_times', default=5, type=int, help='the max length for each agent\'s relative script length (retrival from current script by RAG')

    # TODA LLAMA
    parser.add_argument('--ckpt_dir', default=".llama2_model/Llama-2-13b-hf", type=str,required=False, help='if llama')

    # for log
    parser.add_argument('--output_root_path', default=r'./log_Werewolf', type=str, required=False, help='max_tree_depth')

    args = parser.parse_args()
    agent_num2each_rsl={
        9: 1200,
        8: 1600,
        7: 2000,
        6: 2500,
        5: 3000,
        4: 3500,
    }
    model2ckpt = {
        "gemma7b": ".llama2_model/gemma-7b-it/",
        "llama70b": "/scratch/prj/inf_llmcache/hf_cache/models--meta-llama--Llama-2-70b-chat-hf/snapshots/e1ce257bd76895e0864f3b4d6c7ed3c4cdec93e2/",
        "llama13b": ".llama2_model/Llama-2-13b-hf",
        "llama7b": ".llama2_model/Llama-2-7b-chat-hf",
        "gpt35": "",
        "gpt4": "",
        "vllm":"",
    }
    if args.is_english:
        args.database_root = "./database_english"
        args.script_root = "./english"

    args.ckpt_dir = model2ckpt[args.model_type]
    main()