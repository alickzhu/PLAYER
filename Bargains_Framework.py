import argparse
import json
import os
import glob

from utils.Chat_utils import ChatSession
from utils.RAG import VectorDatabase
from utils.llama import load_llama_3
from utils.gemma import load_gemma_model
from utils.json_tools import json_format
from transformers import AutoTokenizer
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

class Agent:

    def __init__(self, args, name = '', product='', description='', target_price='', model=None, tokenizer=None):
        self.session = ChatSession(args=args, model=model, tokenizer=tokenizer)  # get reply
        self.name = name
        self.product = product  # current script
        self.description = description
        self.target_price = target_price
        self.is_buyer = False
        self.is_seller = False

        if self.name == 'buyer':
            self.is_buyer = True
        else:
            self.is_seller = True


    def speak(self, prompt, evaluate = False, return_prob = False):
        return self.session.get_reply(prompt, evaluate, return_prob)



class ScriptManager:

    def __init__(self, args):
        # TODO
        self.scripts = pd.read_csv('Bargains_Dataset/train.csv', encoding='utf-8-sig')
        self.args = args
        self.product_index = args.product_index
        self.product = self.scripts['title'][self.product_index]  # current script
        self.description = self.scripts['description'][self.product_index]  # current script description =
        self.buyer_target = self.scripts['buyer_target'][self.product_index]  # buyer target
        self.seller_target = self.scripts['seller_target'][self.product_index]  # seller target
        self.relative_price = self.scripts['relative_price'][self.product_index] * self.seller_target  # seller target
        self.character_name_list = ["buyer", "seller"]


class Game:
    def __init__(self, scriptmanager, args):
        # read prompts
        self.args = args
        self.scriptmanager = scriptmanager
        with open(args.prompt_path, 'r', encoding='utf-8') as f:
            self.prompts = yaml.load(f, Loader=yaml.FullLoader)
            f.close()
        with open(args.sensor_path, 'r', encoding='utf-8') as f:
            self.sensors = json.load(f)
            f.close()
        # the number of characters who can ask questions when other characters introduce themselves
        self.agents = [] # store all the agents
        self.chat_history = []  # store all the chat history
        self.chat_dataset = VectorDatabase()
        self.evaluate_history = [[] for _ in range(self.args.eva_times)]

        self.rsl = args.rsl  # the max length for relative script length (retrival from current script by RAG)
        self.each_rsl = args.each_rsl  # the max length for relative script length (retrival from current script by RAG)
        self.rchl = args.rchl  # the max length for relative chat history length (retrival from chat history by RAG)
        self.question_number = args.question_number
        self.chat_history_format = []

        self.initial_game()  # initial game

    def initial_game(self):
        """
        initial game
        """
        character_name_list = ["buyer", "seller"]
        # initial agents
        for character_name in character_name_list:
            if "llama" in self.args.model_type or "gemma" in self.args.model_type:
                self.agents.append(Agent(args=self.args, name=character_name, product=self.scriptmanager.product, description=self.scriptmanager.description, target_price='', model=self.model, tokenizer=self.tokenizer))
            else:
                self.agents.append(Agent(args=self.args, name=character_name, product=self.scriptmanager.product, description=self.scriptmanager.description, target_price=''))
        self.agents[0].target_price=self.scriptmanager.buyer_target
        self.agents[1].target_price=self.scriptmanager.seller_target

        self.system = Agent(args=self.args, name=character_name, product=self.scriptmanager.product, description=self.scriptmanager.description, target_price='')

        self.initial_agent_session()  # initial agent system prompt


    def initial_agent_session(self):
        for i in range(len(self.agents)):
            if self.agents[i].is_buyer:
                self.agents[i].session.system = self.prompts['system_prompt_buyer'].format(
                    BUYER_TARGET_PRICE=self.agents[i].target_price)
            else:
                self.agents[i].session.system = self.prompts['system_prompt_seller'].format(
                    SELLER_TARGET_PRICE=self.agents[i].target_price)
        self.system.session.system = self.prompts['system_system_prompt']

    def add_chat(self, chatitem):
        """
            add chat item to chat history and chat dataset (for RAG)
        """
        self.chat_history.append(chatitem)  # add chat item to chat history
        format_chat_item = self.format_chat_item(chatitem)  # get chat string for RAG
        # self.chat_dataset.add_text(format_chat_item, False)  # store chat string to chat dataset
        self.chat_history_format.append(format_chat_item)  # store chat string to chat history format
        print(format_chat_item)
        print()

    def get_all_history(self):
        return "\n".join(self.chat_history_format)
    def format_chat_item(self, chatitem):
        """
        convert chat item to a string that can be used by RAG
        """

        format_chat_item = self.prompts['combine_chat_all'].format(speaker=chatitem.speaker,listener=chatitem.listener, words=chatitem.words)

        return format_chat_item


    def get_agent(self,name):
        """ get agent by character name"""
        return [i for i in self.agents if i.name == name][0]

    def process_reply(self, parts):

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

        path_excel = os.path.join(path,prefix+str(self.scriptmanager.product_index)+'.xlsx')
        try:
            history.to_excel(path_excel, index=False,  encoding='utf-8-sig')
        except:
            try:
                history.to_excel(path_excel, index=False)
            except:
                history.to_csv(path_excel.replace('.xlsx',".csv"), index=False, encoding='utf-8-sig')
        print('save chat history to excel')


    def evalute_para(self, rsl, rchl, path, prefix = ''):
        self.args.rsl = rsl
        self.args.rchl = rchl
        self.evaluate()
        self.save_evaluate_history(path, prefix)

    def save_args_to_json(self, args, filename):
        args_dict = vars(args)
        with open(filename, 'w') as file:
            json.dump(args_dict, file, indent=4, ensure_ascii=False)

    def mkdir_output_dir(self, args, post = ''):

        args.output_root_path = os.path.join(args.output_root_path, str(args.product_index))

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

