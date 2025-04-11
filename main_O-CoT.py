import argparse
import json
import os
from utils.json_tools import json_format
from tqdm import tqdm
from datetime import datetime
from Game_Framework import Agent, ChatItem, EvaluateItem, Game, ScriptManager

class Game_COT(Game):
    def __init__(self, scriptmanager, args):
        super().__init__(scriptmanager, args)

    def self_introduction_phase(self):
        stage = 'self introduction stage'
        for i in tqdm(range(len(self.agents))):

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

    def summarize_your_goal(self):
        for i in range(len(self.agents)):
            summarize_your_goal_prompt = self.prompts['summarize_your_goal'].replace('{goal}',self.agents[i].acts_goal)
            summarize_your_goal = self.agents[i].speak(summarize_your_goal_prompt)
            summarize_your_goal = json_format(summarize_your_goal)
            summarize_your_goal = [value for key,value in summarize_your_goal.items()]
            self.agents[i].goal_q = summarize_your_goal
    def propose_question_talk_stage(self, i, j, victim, murderer = False):

        current_script = self.agents[i].database.query(self.agents[j].name+','+victim, self.rsl)
        chat_history = self.chat_dataset.query(self.agents[j].name+','+victim, self.rsl)
        summrize = ''
        for k in range(len(self.agents[i].goal_q)):
            ask_yourself_prompt = self.prompts['ask_yourself'].replace('{question}',self.agents[i].goal_q[k]).replace('{current_script}',current_script).replace('{chat_history}',chat_history)
            ask_yourself = self.agents[i].speak(ask_yourself_prompt)
            summrize += "For question {}, your answer is {} \n".format(self.agents[i].goal_q[k], ask_yourself)

        question_propose_prompt = self.prompts['question_propose'].replace('{character}',self.agents[j].name).replace('{question_answer}',summrize).replace('{victim}',victim).replace('{question_number}',self.question_number).replace('{current_script}',current_script).replace('{chat_history}',chat_history)
        question = self.agents[i].speak(question_propose_prompt)

        return question, question_propose_prompt

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

                    if self.agents[i].kill_by_me[x] == 1:
                        question, question_prompt = self.propose_question_talk_stage(i, j, victim, True)
                    else:
                        question, question_prompt = self.propose_question_talk_stage(i, j, victim, False)
                    question = self.process_reply(question)

                    try:
                        self.add_chat(ChatItem(stage, self.agents[i].name, self.agents[j].name, False, question, question_prompt, is_ask=1))
                        # get relay agent
                        relay_agent = self.agents[j]
                        # reply:
                        current_script = relay_agent.database.query(question, self.rsl)
                        chat_history = self.chat_dataset.query(question, self.rchl)

                        if relay_agent.kill_by_me[x] == 1:
                            reply_prompt = self.prompts['open_talk_reply_murder'].format(current_script=current_script,goal=relay_agent.acts_goal,chat_history=chat_history,character=self.agents[i].name,question=question, victim=victim)
                        else:
                            reply_prompt = self.prompts['open_talk_reply'].format(current_script=current_script,goal=relay_agent.acts_goal,chat_history=chat_history,character=self.agents[i].name,question=question)

                        # reply_prompt = self.prompts['open_talk_reply_rap'].format(current_script=current_script,goal=relay_agent.acts_goal,chat_history=chat_history,character=self.agents[i].name,question=question)
                        reply = relay_agent.speak(reply_prompt)
                        reply_pro = self.process_reply(reply)
                        self.add_chat(ChatItem(stage, relay_agent.name, self.agents[i].name, False, reply_pro, reply_prompt, is_ask=0))
                        self.add_chat_database(2)
                    except:
                        continue

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
        self.summarize_your_goal()
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
    game = Game_COT(scriptmanager, args)
    game.mkdir_output_dir(args)
    game.start()

    game.save_history(args.output_root_path)

    path = args.output_root_path

    game.evalute_para(4000, 4000, path)
    # game.save_all_evaluate(path)
    game.save_params(args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # for script
    parser.add_argument('--script_root', default=r"./chinese", type=str,required=False)
    parser.add_argument('--database_root', default=r"./database", type=str,required=False)
    parser.add_argument('--script_name', default="131-罪恶（4人封闭）", type=str, required=False)
    parser.add_argument('--prompt_path', default="./prompts_O-CoT.yaml", type=str,required=False)
    parser.add_argument('--sensor_path', default="./sensors.json", type=str,required=False)
    parser.add_argument('--load_history', default="./log/152-绝命阳光号（4人封闭）_S_1_C_1_T_4_Q_2", type=str,required=False)

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
    parser.add_argument('--output_root_path', default=r'./log_2025/log_O-CoT', type=str, required=False, help='max_tree_depth')

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
        "gpt4": ""
    }
    if args.is_english:
        args.database_root = "./database_english"
        args.script_root = "./english"

    main()