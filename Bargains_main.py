import argparse
import json
import os

from utils.json_tools import json_format
import math
from tqdm import tqdm

from  Bargains_Framework import Agent, ChatItem, ScriptManager, Game

class Game_Player(Game):
    def __init__(self, scriptmanager, args):
        # 初始化父类
        super().__init__(scriptmanager, args)
        self.sensors_ask = []
        self.sensors_pruner = []
        self.suspect_list = []

    def self_introduction_phase(self):
        stage = 'Seller introduction stage'
        for i in range(len(self.agents)):
            if self.agents[i].is_seller:
                intro_prompt = self.prompts['seller_intro'].format(product_name =self.agents[i].product, product_description = self.agents[i].description,target_price = self.agents[i].target_price)
                intro = self.agents[i].speak(intro_prompt)
                self.add_chat(ChatItem(stage, self.agents[i].name, "buyer", False, intro,intro_prompt, is_ask=2))
                # self.add_chat_database(1)

    def propose_question_talk_stage(self):

        chat_history = self.get_all_history()
        summary = []
        sensor_ask = {"sensors":[]}
        for sensor_index in range(len(self.sensors)):
            sensor_single = self.sensors[str(sensor_index)]
            if sensor_single['question_ask'] == 1:
                # choices = sensor_single["choices"]
                choices = ' or '.join(['"'+ item + '"' for item in sensor_single["choices"]])
                sensor_prompt = self.prompts['sensor_ask_bargain'].replace('{chat_history}',chat_history).replace('{sensor}',sensor_single["prompt"]).replace('{choices}',choices)
                sensor_reply = self.agents[0].speak(sensor_prompt)
                sensor_ask["sensors"].append({"sensor":sensor_single["prompt"],"reply":sensor_reply})

                sensor_summary_single = self.prompts["sensor_summary_single_bargain"].format(name=sensor_single["summary"],opinion=sensor_reply)
                summary.append(sensor_summary_single)
        summary_str = '\n'.join(summary)

        self.sensors_ask.append(sensor_ask)
        ask_question_prompt = self.prompts['sensor_question_get_bargains'].replace('{summary}',summary_str).replace('{chat_history}',chat_history).replace('{target}',str(self.agents[0].target_price))
        questions = self.agents[0].speak(ask_question_prompt)
        reply = json_format(questions)['reply']
        return reply, ask_question_prompt

    def check_finish(self):
        chat_history = self.get_all_history()
        prompt = self.prompts['system_check_prompt'].replace('{history}',chat_history)
        finish_check = self.system.speak(prompt)
        reply = json_format(finish_check)
        if int(reply['deal_closed']) == 1:
            self.final_price = float(reply['deal_price'])
            return True
        else:
            self.final_price = None
            return False
    def open_talk(self):
        stage = 'Open talk stage'
        for turn in tqdm(range(self.args.max_turn)):
            if self.args.method == "ours":
                question, question_prompt = self.propose_question_talk_stage()
                self.add_chat(ChatItem(stage, self.agents[0].name, self.agents[1].name, False, question, question_prompt, is_ask=1))
            elif self.args.method == "cot":
                question_prompt = self.prompts['cot_question_bargains'].format(chat_history=self.get_all_history())
                question = self.agents[0].speak(question_prompt)
                self.add_chat(ChatItem(stage, self.agents[0].name, self.agents[1].name, False, question, question_prompt, is_ask=1))
            relay_agent = self.agents[1]
            chat_history = self.get_all_history()
            reply_prompt = self.prompts['open_talk_reply_bargains'].format(chat_history=chat_history)
            reply = relay_agent.speak(reply_prompt)
            reply_pro = self.process_reply(reply)
            self.add_chat(ChatItem(stage, relay_agent.name, self.agents[0].name, False, reply_pro, reply_prompt, is_ask=0))
            is_finish = self.check_finish()
            if is_finish:
                break
        return is_finish, turn
    def start(self):
        self.self_introduction_phase()
        self.is_finish,self.finish_turn = self.open_talk()
        if self.is_finish:
            if self.agents[0].target_price >= self.final_price:
                self.is_success = True
            else:
                self.is_success = False




    def save_args_to_json(self, args, filename):
        args_dict = vars(args)
        with open(filename, 'w') as file:
            json.dump(args_dict, file, indent=4, ensure_ascii=False)

    def mkdir_output_dir(self, args, post = ''):
        args.output_root_path = os.path.join(args.output_root_path, str(args.product_index))

        if not os.path.exists(args.output_root_path):
            os.makedirs(args.output_root_path)
    def save_result(self, output_dir):
        """
        保存成交与胜负信息
        """
        result = {
            "is_success": int(self.is_success) if hasattr(self, 'is_success') else 0,
            "final_price": self.final_price if hasattr(self, 'final_price') and self.final_price is not None else -1,
            "finish_turn": self.finish_turn if hasattr(self, 'finish_turn') else -1,
        }
        with open(os.path.join(output_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

def main():

    """play game"""
    scriptmanager = ScriptManager(args)
    game = Game_Player(scriptmanager, args)
    game.mkdir_output_dir(args)
    game.start()
    game.save_history(args.output_root_path)
    game.save_result(args.output_root_path)


def ablation():
    args.output_root_path = 'logs/log'

    """play game"""
    scriptmanager = ScriptManager(args)
    game = Game(scriptmanager, args)
    game.mkdir_output_dir_time(args)
    game.ablation_turn()
    game.save_all_evaluate(args.output_root_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # for script
    parser.add_argument('--product_index', default=0, type=int,required=False)
    parser.add_argument('--database_root', default=r"./database", type=str,required=False)
    # parser.add_argument('--script_name', default="131-罪恶（4人封闭）", type=str, required=False)
    parser.add_argument('--prompt_path', default="./prompts.yaml", type=str,required=False)
    parser.add_argument('--sensor_path', default="./sensors_bargains.json", type=str,required=False)
    parser.add_argument('--method', default="ours", type=str,required=False, help='[ours, cot,think,werewolf]')

    # for model
    parser.add_argument('--temperature', default=0.8, type=float, help='the diversity of generated text')
    parser.add_argument('--model_type', default="gpt", type=str, required=False, help='[gpt35, gpt4, llama13b]')
    parser.add_argument('--model_name', default="qwen", type=str, required=False, help='[gpt35, gpt4, llama13b]')

    # for game

    parser.add_argument('--rsl', default=4000, type=float, help='the max length for relative script length (retrival from current script by RAG')
    parser.add_argument('--rchl', default=4000, type=float, help='the max length for relative chat history length (retrival from chat history by RAG')
    parser.add_argument('--each_rsl', default=3000, type=float, help='the max length for each agent\'s relative script length (retrival from current script by RAG')
    parser.add_argument('--question_number', default='1', type=str, help='the max length for each agent\'s relative script length (retrival from current script by RAG')
    parser.add_argument('--max_turn', default=10, type=int, help='the max length for each agent\'s relative script length (retrival from current script by RAG')
    parser.add_argument('--constraint', default=1, type=int, help='the max length for each agent\'s relative script length (retrival from current script by RAG')
    parser.add_argument('--sensor', default=1, type=int, help='the max length for each agent\'s relative script length (retrival from current script by RAG')
    parser.add_argument('--ablation', default=0, type=int, help='the max length for each agent\'s relative script length (retrival from current script by RAG')
    parser.add_argument('--evaluate', default=0, type=int, help='the max length for each agent\'s relative script length (retrival from current script by RAG')
    parser.add_argument('--is_english', default=0, type=int, help='the max length for each agent\'s relative script length (retrival from current script by RAG')
    parser.add_argument('--eva_times', default=5, type=int, help='the max length for each agent\'s relative script length (retrival from current script by RAG')

    # TODA LLAMA
    parser.add_argument('--ckpt_dir', default="meta-llama/Llama-3.1-70B-Instruct", type=str,required=False, help='if llama')
    # parser.add_argument('--ckpt_dir', default="meta-llama/Llama-3.3-70B-Instruct", type=str,required=False, help='if llama')
    # parser.add_argument('--ckpt_dir', default="meta-llama/Llama-3.1-8B-Instruct", type=str,required=False, help='if llama')
    # parser.add_argument('--ckpt_dir', default="none", type=str,required=False, help='if llama')
    # parser.add_argument('--ckpt_dir', default="meta-llama/Llama-3.3-70B-Instruct", type=str,required=False, help='if llama')
    # parser.add_argument('--ckpt_dir', default="Qwen/Qwen2.5-72B-Instruct", type=str,required=False, help='if llama')

    # for log
    parser.add_argument('--output_root_path', default=r'./log_bargains/ours', type=str, required=False, help='max_tree_depth')

    args = parser.parse_args()

    main()