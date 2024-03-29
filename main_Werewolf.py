import argparse
import json
import os
import glob
from utils.Chat_utils import ChatSession
from utils.RAG import VectorDatabase
from utils.llama import load_llama_model
from utils.gemma import load_gemma_model
from utils.json_tools import json_format
import yaml
import pandas as pd
import numpy as np
import math
from tqdm import tqdm
from datetime import datetime
import pickle
class ChatItem:
    _index_counter = 0
    def __init__(self, stage='', speaker='', listener='', private=False, words='', prompt='', is_ask=1):
        self.index = ChatItem._index_counter
        ChatItem._index_counter += 1
        self.stage = stage  # the stage of the chat
        self.speaker = speaker  # the speaker of the chat
        self.listener = listener  # the listener of the chat
        self.private = private  # whether the chat is private
        self.words = words  # the content of the chat
        self.is_ask = is_ask  # the prompt of the chat
        self.prompt = prompt  # the prompt of the chat


class EvaluateItem:
    _index_counter = 0
    def __init__(self, tester='', question='', answer='', choices='', is_right='', truth='', value=''):
        self.index = ChatItem._index_counter
        ChatItem._index_counter += 1
        self.tester = tester  # the tester of the question
        self.question = question
        self.choices = choices
        self.truth = truth
        self.answer = answer
        self.is_right = is_right
        self.value = value


class Agent:

    def __init__(self, args, name = '', script='', acts_goal='', final_goal='', secrets='', acts_performance="", is_murderer=False, question=[], model=None, tokenizer=None):
        self.session = ChatSession(args=args, model=model, tokenizer=tokenizer)  # get reply
        self.name = name
        self.script = script  # current script
        self.acts_goal = acts_goal
        self.final_goal = final_goal
        self.questions = question
        self.acts_performance = acts_performance
        self.secrets = secrets
        self.is_murderer = is_murderer  # murderer or civilian
        self.database = VectorDatabase()
        self.summary = ''
        self.character_suspect = []

    def speak(self, prompt, evaluate = False):
        reply = self.session.get_reply(prompt, evaluate)
        return reply

class ScriptManager:

    def __init__(self, args):
        # {character：script(include goal, script, performance...)}
        self.agents_scripts = {}
        # {character：evaluate questions}
        # self.agents_evaluate = {}
        self.agents_final_result = {}
        # script parameters
        para_file_path = os.path.join(args.script_root, args.script_name, 'json', 'script_info.json')
        para = self.read_json(para_file_path)
        self.agent_num = para["agent_num"]  # script character number
        self.character_name_list = para["character_name"]  # script character name
        self.name2index = {name:str(i) for i, name in enumerate(self.character_name_list)}
        self.index2name = {str(i):name for i, name in enumerate(self.character_name_list)}
        self.script_name = para["script_name"]  # script  name
        self.open_discuss_rounds = para["open_discuss_rounds"]  # open talk rounds
        self.acts_num = para["acts_num"]  # acts number
        self.keep_asking = para["keep_asking"]  # continue asking number
        self.privacy_discuss_rounds = para["privacy_discuss_rounds"]  # privacy talk rounds
        # read each character's script
        for name in self.character_name_list:
            script_file_path = os.path.join(args.script_root, args.script_name, 'json', name + '.json')
            script_data = self.read_json(script_file_path)
            self.agents_scripts[name] = script_data
        for name in self.character_name_list:
            final_result_file_path = os.path.join(args.script_root, args.script_name, 'final_result', name + '.csv')
            final_result = pd.read_csv(final_result_file_path, encoding='utf-8-sig')
            self.agents_final_result[name] = final_result
        self.agents_final_result['GM'] = pd.read_csv(os.path.join(args.script_root, args.script_name, 'final_result', 'GM.csv'), encoding='utf-8-sig')

    def update_script(self, agent, n):
        """
        update agent's script according to the acts number
        """
        if n > self.agent_num - 1:
            print('error')
            return
        name = agent.name
        agent.is_murderer = True if self.agents_scripts[name]["is_murderer"] == 1 else False
        agent.character_name_list = [i for i in self.character_name_list if i != name]

        agent.script = '\n'.join(self.agents_scripts[name]["script"][:n + 1])
        agent.acts_goal = '\n'.join(self.agents_scripts[name]["acts_goal"][:n + 1])
        # agent.acts_performance = self.agents_scripts[name]["acts_performance"][n]
        # agent.secrets = self.agents_scripts[name]["secrets"]
        agent.victims = self.agents_scripts[name]["victims"]
        agent.kill_by_me = self.agents_scripts[name]["kill_by_me"]
        # agent.questions = self.agents_scripts[name]["questions"][n]
        if n == self.agent_num - 1:
            agent.final_goal = self.agents_scripts[name]["final_goal"]
        if agent.character_suspect == []:
            agent.character_suspect = [[i for i in self.character_name_list if i != name]]*len(agent.victims)
        return agent


    def read_json(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    def get_script(self, agent_name):
        return self.agents_scripts.get(agent_name, {})


class Game:
    def __init__(self, scriptmanager, args):
        # read prompts
        self.args = args
        with open(args.prompt_path, 'r', encoding='utf-8') as f:
            self.prompts = yaml.load(f, Loader=yaml.FullLoader)
            f.close()
        with open(args.sensor_path, 'r', encoding='utf-8') as f:
            self.sensors = json.load(f)
            f.close()
        # the number of characters who can ask questions when other characters introduce themselves
        self.intro_question_num = 2
        self.script_manager = scriptmanager
        self.agents = [] # store all the agents
        self.GM_agent = Agent(args=self.args) # store all the script
        self.current_stage = None  # current stage
        self.chat_history = []  # store all the chat history
        self.chat_dataset = VectorDatabase()
        self.evaluate_history = []

        self.rsl = args.rsl  # the max length for relative script length (retrival from current script by RAG)
        self.each_rsl = args.each_rsl  # the max length for relative script length (retrival from current script by RAG)
        self.rchl = args.rchl  # the max length for relative chat history length (retrival from chat history by RAG)
        self.question_number = args.question_number

        if "llama" in args.model_type:
            self.model, self.tokenizer = load_llama_model(args.ckpt_dir)
        elif "gemma" in args.model_type:
            self.model, self.tokenizer = load_gemma_model(args.ckpt_dir)

        self.initial_game()  # initial game

    def initial_game(self):
        """
        initial game
        """
        character_name_list = self.script_manager.character_name_list
        # initial agents
        for character_name in character_name_list:
            if "llama" in self.args.model_type or "gemma" in self.args.model_type:
                self.agents.append(Agent(args=self.args, name=character_name, model=self.model, tokenizer=self.tokenizer))
            else:
                self.agents.append(Agent(args=self.args, name=character_name))
        self.update_script(0)  # initial agent script(get act 0 script)
        self.initial_agent_session()  # initial agent system prompt

    def add_chat(self, chatitem):
        """
            add chat item to chat history and chat dataset (for RAG)
        """
        self.chat_history.append(chatitem)  # add chat item to chat history
        format_chat_item = self.format_chat_item(chatitem)  # get chat string for RAG
        # self.chat_dataset.add_text(format_chat_item, False)  # store chat string to chat dataset
        print(format_chat_item)
    def add_chat_database(self, last_n):
        chat = ''
        for i in range(last_n, 0, -1):
            chat += self.format_chat_item(self.chat_history[-i]) + '\n'  # get chat string for RAG
        self.chat_dataset.add_text(chat, False)  # store chat string to chat dataset

    def format_chat_item(self, chatitem):
        """
        convert chat item to a string that can be used by RAG
        """
        if chatitem.is_ask == 1:
            format_chat_item = self.prompts['combine_chat_ask'].format(speaker=chatitem.speaker,listener=chatitem.listener, words=chatitem.words)
        elif chatitem.is_ask == 2:
            format_chat_item = self.prompts['combine_chat_all'].format(speaker=chatitem.speaker,listener=chatitem.listener, words=chatitem.words)
        else:
            format_chat_item = self.prompts['combine_chat_answer'].format(speaker=chatitem.speaker,listener=chatitem.listener, words=chatitem.words)

        return format_chat_item


    def get_chat_history(self, name):
        """
        This function has been deprecated.
        """
        chat_history_list = []
        for chatitem in self.chat_history:
            if chatitem.private == True and chatitem.listener == name:
                chat_history_list.append(self.format_chat_item(chatitem))
            if chatitem.private == False:
                chat_history_list.append(self.format_chat_item(chatitem))
        return ', '.join(chat_history_list)

    def update_script(self, n):
        """
        update agent's script according to the acts number
        add the updated script to the script dataset
        """
        all_scrpt = ''
        for i in range(len(self.agents)):
            self.agents[i] = self.script_manager.update_script(self.agents[i], n)
            all_scrpt += self.agents[i].script
            file_dir = os.path.join(args.database_root, args.script_name.split("-")[0])
            file_name_faiss = os.path.join(args.database_root, args.script_name.split("-")[0],  ''.join([self.script_manager.name2index[self.agents[i].name],'_act_',str(n),'.faiss']))
            file_name_doc = os.path.join(args.database_root, args.script_name.split("-")[0],  ''.join([self.script_manager.name2index[self.agents[i].name],'_act_',str(n),'.pickle']))
            if os.path.exists(file_name_faiss):
                self.agents[i].database.load_faiss(file_name_faiss)
                self.agents[i].database.load_texts(file_name_doc)

            else:
                self.agents[i].database.reset()
                self.agents[i].database.add_text(self.agents[i].script)
                if not os.path.exists(file_dir):
                    os.makedirs(file_dir)
                self.agents[i].database.save_faiss(file_name_faiss)
                self.agents[i].database.save_texts(file_name_doc)

    def initial_agent_session(self):
        """initial agent system prompt according to the role (civilian or murderer)"""
        self.GM_agent.session.system = self.prompts['system_prompt_gm']
        for i in range(len(self.agents)):
            if self.agents[i].is_murderer:
                self.agents[i].session.system = self.prompts['system_prompt_murderer'].format(
                    character_name=self.agents[i].name,
                    character_name_list=', '.join(self.agents[i].character_name_list))
            else:
                self.agents[i].session.system = self.prompts['system_prompt_civilian'].format(
                    character_name=self.agents[i].name,
                    character_name_list=', '.join(self.agents[i].character_name_list))

    def get_agent(self,name):
        """ get agent by character name"""
        return [i for i in self.agents if i.name == name][0]

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

    def save_history(self,path,prefix = ''):
        history_list = []
        for chatitem in self.chat_history:
            history_list.append({
                "index" : chatitem.index,
                "stage" : chatitem.stage,
                "speaker" : chatitem.speaker,
                "listener" : chatitem.listener,
                # "private" : chatitem.private,
                "words" : chatitem.words,
                "prompt" : chatitem.prompt
            })
        history = pd.DataFrame(history_list)

        path_excel = os.path.join(path,prefix+self.script_manager.script_name+'.xlsx')
        history.to_excel(path_excel, index=False,  encoding='utf-8-sig')
        print('save chat history to excel')

        file_name_faiss = os.path.join(path, prefix+self.script_manager.script_name+'chat_history.faiss')
        file_name_doc = os.path.join(path, prefix+self.script_manager.script_name+'chat_history.pickle')
        self.chat_dataset.save_faiss(file_name_faiss)
        self.chat_dataset.save_texts(file_name_doc)

    def load_history(self, path):
        """
        load chat history from excel
        """

        file_name_faiss = os.path.join(path, self.script_manager.script_name+'chat_history.faiss')
        file_name_doc = os.path.join(path, self.script_manager.script_name+'chat_history.pickle')
        if os.path.exists(file_name_faiss):
            self.GM_agent.database.load_faiss(file_name_faiss)
            self.GM_agent.database.load_texts(file_name_doc)



    def save_evaluate_history(self,path,prefix = ''):
        path = os.path.join(path, 'evaluate')
        if not os.path.exists(path):
            os.makedirs(path)
        history_list = []
        for chatitem in self.evaluate_history:
            if chatitem.value == "a":
                chatitem.value = 10
            elif chatitem.value == "b":
                chatitem.value = 5
            elif chatitem.value == "c":
                chatitem.value = 2
            history_list.append({
                "tester" : chatitem.tester,
                "value" : chatitem.value,
                "question" : chatitem.question,
                "choices" : chatitem.choices.replace('\n',' '),
                "truth" : chatitem.truth,
                "answer" : chatitem.answer,
                "is_right" : chatitem.is_right

            })
        history = pd.DataFrame(history_list)
        type_a = history[history['value'] == 10]['is_right'].mean()
        type_b = history[history['value'] == 5]['is_right'].mean()
        type_c = history[history['value'] == 2]['is_right'].mean()
        accuracy = history['is_right'].mean()

        correct_score = history[history['is_right']==1]['value'].sum()
        total_score = history['value'].sum()
        score = correct_score / total_score
        new_row = pd.DataFrame({'is_right': [score],
                                "answer":str(correct_score) + '/' + str(total_score),
                                "question":type_a,
                                "choices":type_b,
                                "truth":type_c,
                                })
        # 将新行追加到DataFrame的末尾
        history = pd.concat([history, new_row], ignore_index=True)
        if prefix == 'GM_':
            appendix = '_eva_{}.xlsx'.format(self.each_rsl)
        else:
            appendix = '_script_{}_chat_{}.xlsx'.format(self.args.rsl,self.args.rchl)
        path_excel = os.path.join(path,prefix+self.script_manager.script_name+appendix)
        history.to_excel(path_excel, index=False,  encoding='utf-8-sig')
        print(path_excel)
        return path

    def evaluate(self):
        """
        evaluate the answer of the question
        TODA
        """
        self.evaluate_history = []
        for j in tqdm(range(len(self.agents))):
            data = self.script_manager.agents_final_result[self.agents[j].name]
            for i in tqdm(range(len(data))):
                # for i in range(len(data)):
                question = data['question'][i]
                if pd.isna(data['c'][i]):
                    choices = '\nA: '+ data['a'][i] + '\n' + 'B: '+ data['b'][i] + '\n'
                elif pd.isna(data['d'][i]):
                    choices = '\nA: '+ data['a'][i] + '\n' + 'B: '+ data['b'][i] + '\n' + 'C: '+ data['c'][i] + '\n'
                elif pd.isna(data['e'][i]):
                    choices = '\nA: '+ data['a'][i] + '\n' + 'B: '+ data['b'][i] + '\n' + 'C: '+ data['c'][i] + '\n' + 'D: '+ data['d'][i] + '\n'
                else:
                    choices = '\nA: '+ data['a'][i] + '\n' + 'B: '+ data['b'][i] + '\n' + 'C: '+ data['c'][i] + '\n' + 'D: '+ data['d'][i] + '\n' + 'E: '+ data['e'][i] + '\n'
                truth = data['truth'][i]
                current_script = self.agents[j].database.query(question, self.rsl)
                chat_history = self.chat_dataset.query(question, self.rchl)
                if data['type'][i] == 'a':
                    question_prompt = self.prompts['evaluate_ask_rap'].replace('{current_script}',current_script).replace('{chat_history}',chat_history).replace('{question}',question).replace('{choices}',choices)
                else:
                    question_prompt = self.prompts['evaluate_ask_rap_multi'].replace('{current_script}',current_script).replace('{chat_history}',chat_history).replace('{question}',question).replace('{choices}',choices)
                answer = self.agents[j].speak(question_prompt,True)
                is_right = 0
                format_answer = json_format(answer)
                try:
                    if data['type'][i] == 'b':
                        format_answer = format_answer['answer'].lower()
                        if truth.lower() in format_answer:
                            is_right = 1
                    else:
                        format_answer = format_answer['answer'].lower()
                        if format_answer in truth.lower():
                            is_right = 1
                except:
                    pass
                self.evaluate_history.append(EvaluateItem(tester=self.agents[j].name, question = question, choices=choices, answer=answer,truth = truth, is_right=is_right, value=data['value'][i]))

    def evaluate_gm(self, version=0):
        self.evaluate_history = []

        data = self.script_manager.agents_final_result['GM']
        for i in tqdm(range(len(data))):
            # for i in range(len(data)):
            question = data['question'][i]
            if pd.isna(data['c'][i]):
                choices = '\nA: '+ data['a'][i] + '\n' + 'B: '+ data['b'][i] + '\n'
            elif pd.isna(data['d'][i]):
                choices = '\nA: '+ data['a'][i] + '\n' + 'B: '+ data['b'][i] + '\n' + 'C: '+ data['c'][i] + '\n'
            elif pd.isna(data['e'][i]):
                choices = '\nA: '+ data['a'][i] + '\n' + 'B: '+ data['b'][i] + '\n' + 'C: '+ data['c'][i] + '\n' + 'D: '+ data['d'][i] + '\n'
            else:
                choices = '\nA: '+ data['a'][i] + '\n' + 'B: '+ data['b'][i] + '\n' + 'C: '+ data['c'][i] + '\n' + 'D: '+ data['d'][i] + '\n' + 'E: '+ data['e'][i] + '\n'
            truth = data['truth'][i]
            # current_script = self.GM_agent.database.query(question, self.rsl)
            if version == 0:
                self.version = 'average'
                relative_script = ''
                for j in range(len(self.agents)):
                    name = self.agents[j].name
                    current_script = self.agents[j].database.query(question, self.each_rsl)
                    relative_script +="The information in {}'s script related to this question is: \n {} \n\n".format(name, current_script)

            if data['type'][i] == 'a':
                question_prompt = self.prompts['evaluate_ask_rap_gm'].replace('{current_script}',relative_script).replace('{question}',question).replace('{choices}',choices)
            else:
                question_prompt = self.prompts['evaluate_ask_rap_multi_gm'].replace('{current_script}',relative_script).replace('{question}',question).replace('{choices}',choices)
            answer = self.GM_agent.speak(question_prompt, True)
            is_right = 0
            format_answer = json_format(answer)
            try:
                format_answer = format_answer['answer'].lower()
                if data['type'][i] == 'b':
                    if truth.lower() in format_answer:
                        is_right = 1
                else:
                    if format_answer[0] in truth.lower():
                        is_right = 1
            except:
                pass
            self.evaluate_history.append(EvaluateItem(tester="GM", question = question, choices=choices, answer=answer,truth = truth, is_right=is_right, value=data['value'][i]))


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
        # now = datetime.now()
        # time_string = now.strftime("%Y%m%d%H%M%S")
        # args.output_root_path = os.path.join(args.output_root_path, args.script_name+time_string)
        # post = "_S_{}_C_{}_T_{}_Q_{}".format(args.sensor, args.constraint, args.max_turn, args.question_number)

        args.output_root_path = os.path.join(args.output_root_path, args.script_name+post)

        if not os.path.exists(args.output_root_path):
            os.makedirs(args.output_root_path)

    def save_params(self, args):
        input_token = []
        output_token = []
        input_token_eva = []
        output_token_eva = []

        for i in range(len(self.agents)):
            input_token += self.agents[i].session.input_token
            output_token += self.agents[i].session.output_token
            input_token_eva += self.agents[i].session.input_token_eva
            output_token_eva += self.agents[i].session.output_token_eva
        args.input_token = sum(input_token)
        args.output_token = sum(output_token)
        args.input_price = args.input_token * 0.0005 / 1000
        args.output_price = args.output_token * 0.0015 / 1000

        args.input_token_eva = sum(input_token_eva)
        args.output_token_eva = sum(output_token_eva)
        args.input_eva_price = args.input_token_eva * 0.0005 / 1000
        args.output_eva_price = args.output_token_eva * 0.0015 / 1000

        args.play_price = args.input_price + args.output_price
        args.eva_price = args.input_eva_price + args.output_eva_price

        self.save_args_to_json(args, os.path.join(args.output_root_path, 'args.json'))

    def mkdir_output_dir_time(self, args):
        now = datetime.now()
        time_string = now.strftime("%Y%m%d%H%M%S")
        args.output_root_path = os.path.join(args.output_root_path, args.script_name+time_string)
        self.args.output_root_path = args.output_root_path
        # post = "_S_{}_C_{}_T_{}_Q_{}".format(args.sensor, args.constraint, args.max_turn, args.question_number)
        # args.output_root_path = os.path.join(args.output_root_path, args.script_name+post)

        if not os.path.exists(args.output_root_path):
            os.makedirs(args.output_root_path)
        self.save_args_to_json(args, os.path.join(args.output_root_path, 'args.json'))

    def save_all_evaluate(self, path):

        path = os.path.join(path, 'evaluate')
        xlsx_files = glob.glob(os.path.join(path, "*.xlsx"))
        data = []
        for file in xlsx_files:
            df = pd.read_excel(file)
            last_value = df["is_right"].iloc[-1]
            detail_value = df["answer"].iloc[-1]
            type_a = df["question"].iloc[-1]
            type_b = df["choices"].iloc[-1]
            type_c = df["truth"].iloc[-1]

            filename = os.path.basename(file)
            data.append((filename,type_a, type_b,type_c,detail_value, last_value, len(df)-1))
        result_df = pd.DataFrame(data, columns=["filename", "type_a","type_b","type_c","detail_value", "score","length"])
        result_df = result_df.sort_values(by='filename')
        accuracy = result_df[~result_df['filename'].str.contains('GM') & ~result_df['filename'].str.contains('no_play')]['score'].mean()
        accuracy_a = result_df[~result_df['filename'].str.contains('GM') & ~result_df['filename'].str.contains('no_play')]['type_a'].mean()
        accuracy_b = result_df[~result_df['filename'].str.contains('GM') & ~result_df['filename'].str.contains('no_play')]['type_b'].mean()
        accuracy_c = result_df[~result_df['filename'].str.contains('GM') & ~result_df['filename'].str.contains('no_play')]['type_c'].mean()
        new_row = pd.DataFrame({'score': [accuracy],'type_a': [accuracy_a],'type_b': [accuracy_b],'type_c': [accuracy_c]})
        result_df = pd.concat([result_df, new_row], ignore_index=True)
        import numpy as np
        def format_significant(x):
            if isinstance(x, float):
                return np.format_float_positional(x, precision=3, unique=False, fractional=False, trim='k')
            return x

        result_df = result_df.applymap(format_significant)
        output_csv_path = os.path.join(path, "data_summary.csv")
        result_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

def evalute_para(game,  path, prefix = ''):

    game.evaluate()
    game.save_evaluate_history(path, prefix)

def main():

    """play game"""
    scriptmanager = ScriptManager(args)
    game = Game(scriptmanager, args)
    game.mkdir_output_dir(args)
    game.start()

    game.save_history(args.output_root_path)

    path = args.output_root_path

    evalute_para(game, path, "0")
    evalute_para(game, path, "1")
    evalute_para(game, path, "2")

    game.save_all_evaluate(path)
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

    parser.add_argument('--load_history', default="/scratch/prj/lmrep/qinglin/Murder_Mystery_Game/log_Werewolf/134-致命喷泉（4人封闭）Werewolf", type=str,required=False)

    # for model
    parser.add_argument('--temperature', default=0.9, type=float, help='the diversity of generated text')
    parser.add_argument('--model_type', default="gpt35", type=str, required=False, help='[gpt35, gpt4, llama13b]')

    # for game

    parser.add_argument('--rsl', default=4000, type=float, help='the max length for relative script length (retrival from current script by RAG')
    parser.add_argument('--rchl', default=4000, type=float, help='the max length for relative chat history length (retrival from chat history by RAG')
    parser.add_argument('--each_rsl', default=3000, type=float, help='the max length for each agent\'s relative script length (retrival from current script by RAG')
    parser.add_argument('--question_number', default='1', type=str, help='the max length for each agent\'s relative script length (retrival from current script by RAG')
    parser.add_argument('--max_turn', default=3, type=int, help='the max length for each agent\'s relative script length (retrival from current script by RAG')
    parser.add_argument('--constraint', default=1, type=int, help='the max length for each agent\'s relative script length (retrival from current script by RAG')
    parser.add_argument('--sensor', default=1, type=int, help='the max length for each agent\'s relative script length (retrival from current script by RAG')
    parser.add_argument('--is_english', default=1, type=int, help='the max length for each agent\'s relative script length (retrival from current script by RAG')

    # TODA LLAMA
    parser.add_argument('--ckpt_dir', default="/scratch/prj/lmrep/llama2_model/Llama-2-13b-hf", type=str,required=False, help='if llama')

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
        "gemma7b": "/scratch/prj/lmrep/llama2_model/gemma-7b-it/",
        "llama70b": "/scratch/prj/inf_llmcache/hf_cache/models--meta-llama--Llama-2-70b-chat-hf/snapshots/e1ce257bd76895e0864f3b4d6c7ed3c4cdec93e2/",
        "llama13b": "/scratch/prj/lmrep/llama2_model/Llama-2-13b-hf",
        "llama7b": "/scratch/prj/lmrep/llama2_model/Llama-2-7b-chat-hf",
        "gpt35": "",
        "gpt4": ""
    }
    if args.is_english:
        args.database_root = "./database_english"
        args.script_root = "./english"

    args.ckpt_dir = model2ckpt[args.model_type]
    main()
