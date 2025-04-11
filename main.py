import argparse
import json
import os
from utils.json_tools import json_format
import math
from tqdm import tqdm
from  Game_Framework import Agent, ChatItem, ScriptManager, Game

class Game_Player(Game):
    def __init__(self, scriptmanager, args):
        super().__init__(scriptmanager, args)
        self.sensors_ask = []
        self.sensors_pruner = []
        self.suspect_list = []


    def self_introduction_phase(self):
        stage = 'self introduction stage'
        for i in tqdm(range(len(self.agents))):
            if not args.is_english:

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

    def find_most_suspicion(self, i, x, victim, murderer = False, multi = True):
        postfix = ''
        if murderer:
            postfix = '_murder'
        summary_list = []
        for j in range(len(self.agents)):
            sensor_pruner = {"a":self.agents[i].name, "b":self.agents[j].name,"sensors":[]}
            if i == j:
                continue

            current_script = self.agents[i].database.query(self.agents[j].name+','+victim, self.rsl)
            chat_history = self.chat_dataset.query(self.agents[j].name+','+victim, self.rsl)
            summary_single = []
            for sensor_index in range(len(self.sensors)):
                sensor_single = self.sensors[str(sensor_index)]
                if sensor_single['constraint'] == 1:
                    choices = ' or '.join(['"'+ item + '"' for item in sensor_single["choices"]])
                    sensor_prompt = self.prompts['sensor_prompt'].replace('{sensor}',sensor_single["prompt"]).replace('{choices}',choices)
                    summary_single.append(sensor_prompt)

            sensor_prompt = self.prompts['sensor_ask_once'+postfix].replace('{current_script}',current_script).replace('{chat_history}',chat_history).replace('{character}',self.agents[j].name).replace('{victim}',victim).replace('{sensor_prompt}','\n'.join(summary_single))
            sensor_reply = self.agents[i].speak(sensor_prompt)
            sensor_pruner["sensors"].append(sensor_reply)
            summary_str = "About {character}:\n {summary}".format(character = self.agents[j].name, summary = sensor_reply)
            summary_list.append(summary_str)
            self.sensors_pruner.append(sensor_pruner)

        summary = '\n'.join(summary_list)
        character_suspect = str(self.agents[i].character_suspect[x])
        choices ="[" +", ".join([f"{chr(65 + i)}. {name}" for i, name in enumerate(self.agents[i].character_suspect[x])])+"]"

        ask_question_prompt = self.prompts['s_constraint_choice'+postfix].replace('{character}',self.agents[j].name).replace('{summary}',summary).replace('{character_suspect}',character_suspect).replace('{victim}',victim).replace('{murder_choices}',choices)
        update_suspect, top_logprobs = self.agents[i].speak(ask_question_prompt, return_prob=True)


        target_tokens = [chr(65 + i) for i in range(len(self.agents[i].character_suspect[x]))]
        result_logprobs = [next((item.logprob for item in top_logprobs if item.token == token), 0) for token in target_tokens]
        normalized_probs = [math.exp(lp) / sum(math.exp(lp) for lp in result_logprobs) for lp in result_logprobs]

        format_update_suspect = json_format(update_suspect)
        self.suspect_list.append([self.agents[i].name, victim, format_update_suspect])
        return format_update_suspect, ask_question_prompt

    def constraints_space(self):
        for i in tqdm(range(len(self.agents)), desc='constraints_space'):
            for x,victim in enumerate(self.agents[i].victims):
                # get question
                if self.agents[i].kill_by_me[x] == 1:
                    format_update_suspect, ask_question_prompt = self.find_most_suspicion(i, x, victim, True)
                else:
                    format_update_suspect, ask_question_prompt = self.find_most_suspicion(i, x, victim, False)
                try:
                    self.agents[i].character_suspect[x] = format_update_suspect['suspicion']
                except:
                    continue

    def propose_question_talk_stage(self, i, j, victim, murderer = False):

        postfix = ''
        if murderer:
            postfix = '_murder'
        if self.args.sensor:
            current_script = self.agents[i].database.query(self.agents[j].name+','+victim, self.rsl)
            chat_history = self.chat_dataset.query(self.agents[j].name+','+victim, self.rsl)
            summary = []
            sensor_ask = {"a":self.agents[i].name, "b":self.agents[j].name, "victim":victim,"sensors":[]}
            for sensor_index in range(len(self.sensors)):
                sensor_single = self.sensors[str(sensor_index)]
                if sensor_single['question_ask'] == 1:
                    # choices = sensor_single["choices"]
                    choices = ' or '.join(['"'+ item + '"' for item in sensor_single["choices"]])
                    sensor_prompt = self.prompts['sensor_ask'+postfix].replace('{current_script}',current_script).replace('{chat_history}',chat_history).replace('{character}',self.agents[j].name).replace('{victim}',victim).replace('{sensor}',sensor_single["prompt"]).replace('{choices}',choices)
                    sensor_reply = self.agents[i].speak(sensor_prompt)
                    sensor_ask["sensors"].append({"sensor":sensor_single["prompt"],"reply":sensor_reply})

                    sensor_summary_single = self.prompts["sensor_summary_single"].format(name=sensor_single["summary"],character=self.agents[j].name,opinion=sensor_reply)
                    summary.append(sensor_summary_single)
            summary_str = '\n'.join(summary)
        else:
            summary_str = ''
        self.sensors_ask.append(sensor_ask)
        current_script = self.agents[i].database.query(self.agents[j].name+','+victim, self.rsl)
        chat_history = self.chat_dataset.query(self.agents[j].name+','+victim, self.rsl)
        ask_question_prompt = self.prompts['sensor_question_get'+postfix].replace('{character}',self.agents[j].name).replace('{summary}',summary_str).replace('{victim}',victim).replace('{question_number}',self.question_number).replace('{current_script}',current_script).replace('{chat_history}',chat_history)
        questions = self.agents[i].speak(ask_question_prompt)
        format_question = json_format(questions)
        return format_question, ask_question_prompt

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
                        questions, question_prompt = self.propose_question_talk_stage(i, j, victim, True)
                    else:
                        questions, question_prompt = self.propose_question_talk_stage(i, j, victim, False)

                    try:
                        for key, question in questions.items():
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
                            # if reply!=reply_pro:
                            #     print(reply)
                            self.add_chat(ChatItem(stage, relay_agent.name, self.agents[i].name, False, reply_pro, reply_prompt, is_ask=0))
                            self.add_chat_database(2)
                    except:
                        continue


    def start(self):
        self.self_introduction_phase()
        count = 0
        while count < self.args.max_turn:
            self.open_talk()
            if count + 1 < self.args.max_turn:
                if self.args.constraint:
                    self.constraints_space()
            count += 1

    def ablation_turn(self):
        self.self_introduction_phase()
        count = 0
        while count < self.args.max_turn:
            self.open_talk()

            self.evaluate()
            self.args.rsl = 5000
            self.args.rchl = 5000
            self.evaluate()
            self.save_evaluate_history(self.args.output_root_path, 'turn_'+str(count)+'_0')
            self.save_evaluate_history(self.args.output_root_path, 'turn_'+str(count)+'_1')
            self.save_evaluate_history(self.args.output_root_path, 'turn_'+str(count)+'_2')
            self.save_history(self.args.output_root_path, 'turn_'+str(count))

            if count + 1 == self.args.max_turn:
                continue
            if self.args.constraint:
                self.constraints_space()
            print(count)
            count += 1

    def save_args_to_json(self, args, filename):
        args_dict = vars(args)
        with open(filename, 'w') as file:
            json.dump(args_dict, file, indent=4, ensure_ascii=False)

    def mkdir_output_dir(self, args, post = ''):
        # now = datetime.now()
        # time_string = now.strftime("%Y%m%d%H%M%S")
        # args.output_root_path = os.path.join(args.output_root_path, args.script_name+time_string)
        # post = "_S_{}_C_{}_T_{}_Q_{}".format(args.sensor, args.constraint, args.max_turn, args.question_number)

        args.output_root_path = os.path.join(args.output_root_path, args.script_name+post)

        if not os.path.exists(args.output_root_path):
            os.makedirs(args.output_root_path)


def main():

    """play game"""
    scriptmanager = ScriptManager(args)
    game = Game_Player(scriptmanager, args)
    game.mkdir_output_dir(args)
    game.start()
    #
    game.save_history(args.output_root_path)

    path = args.output_root_path

    game.evalute_para(4000, 4000, path)
    # evalute_para(game, 5000, 5000,path, "1")
    # evalute_para(game, 5000, 5000,path, "2")

    game.save_all_evaluate(path)
    game.save_params(args)

def ablation():
    args.output_root_path = './log'

    """play game"""
    scriptmanager = ScriptManager(args)
    game = Game(scriptmanager, args)
    game.mkdir_output_dir_time(args)
    game.ablation_turn()

    game.save_all_evaluate(args.output_root_path)

def evalute_from_history():

    scriptmanager = ScriptManager(args)
    game = Game_Player(scriptmanager, args)
    game.load_history(args.load_history)
    path = args.load_history
    game.evalute_para(4000, 4000, path)
    game.save_all_evaluate(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # for script
    parser.add_argument('--script_root', default=r"./chinese", type=str,required=False)
    parser.add_argument('--database_root', default=r"./database", type=str,required=False)
    parser.add_argument('--script_name', default="131-罪恶（4人封闭）", type=str, required=False)
    parser.add_argument('--prompt_path', default="./prompts.yaml", type=str,required=False)
    parser.add_argument('--sensor_path', default="./sensors.json", type=str,required=False)

    parser.add_argument('--load_history', default="./log_human/rc/ours/131-罪恶（4人封闭）", type=str,required=False)

    # for model
    parser.add_argument('--temperature', default=0.8, type=float, help='the diversity of generated text')
    parser.add_argument('--model_type', default="vllm", type=str, required=False, help='[gpt35, gpt4, llama13b]')
    parser.add_argument('--model_name', default="qwen", type=str, required=False, help='[gpt35, gpt4, llama13b]')

    # for game
    parser.add_argument('--rsl', default=4000, type=float, help='the max length for relative script length (retrival from current script by RAG')
    parser.add_argument('--rchl', default=4000, type=float, help='the max length for relative chat history length (retrival from chat history by RAG')
    parser.add_argument('--each_rsl', default=3000, type=float, help='the max length for each agent\'s relative script length (retrival from current script by RAG')
    parser.add_argument('--question_number', default='1', type=str, help='the max length for each agent\'s relative script length (retrival from current script by RAG')
    parser.add_argument('--max_turn', default=3, type=int, help='the max length for each agent\'s relative script length (retrival from current script by RAG')
    parser.add_argument('--constraint', default=1, type=int, help='the max length for each agent\'s relative script length (retrival from current script by RAG')
    parser.add_argument('--sensor', default=1, type=int, help='the max length for each agent\'s relative script length (retrival from current script by RAG')
    parser.add_argument('--ablation', default=0, type=int, help='the max length for each agent\'s relative script length (retrival from current script by RAG')
    parser.add_argument('--evaluate', default=0, type=int, help='the max length for each agent\'s relative script length (retrival from current script by RAG')
    parser.add_argument('--is_english', default=0, type=int, help='the max length for each agent\'s relative script length (retrival from current script by RAG')
    parser.add_argument('--eva_times', default=5, type=int, help='the max length for each agent\'s relative script length (retrival from current script by RAG')

    # TODA LLAMA
    parser.add_argument('--ckpt_dir', default="meta-llama/Llama-3.1-70B-Instruct", type=str,required=False, help='if llama')

    # for log
    parser.add_argument('--output_root_path', default=r'./log_tacl/ours', type=str, required=False, help='max_tree_depth')

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
        "llama3_8b": "meta-llama/Llama-3.1-8B-Instruct",
        "llama3_70b": "meta-llama/Llama-3.1-70B-Instruct",
        "qwen_72b": "Qwen/Qwen2.5-72B-Instruct",
        "llama70b": "/scratch/prj/inf_llmcache/hf_cache/models--meta-llama--Llama-2-70b-chat-hf/snapshots/e1ce257bd76895e0864f3b4d6c7ed3c4cdec93e2/",
        "llama13b": ".llama2_model/Llama-2-13b-hf",
        "llama7b": ".llama2_model/Llama-2-7b-chat-hf",
        "gpt35": "",
        "gpt4": ""
    }
    if args.is_english:
        args.database_root = "./database_english"
        args.script_root = "./english"
    if args.ablation:
        print('ablation begin',args.max_turn,args.question_number)
        ablation()
    elif args.evaluate:
        print('evalute begin')
        evalute_from_history()
    else:
        main()