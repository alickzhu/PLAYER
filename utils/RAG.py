import time

import numpy as np
import faiss  # 确保已安装faiss库
import openai

import re
import tiktoken
import pickle
import os
from openai import OpenAI
os.environ['OPENAI_API_KEY'] = ""
client = OpenAI()
import random
from pathlib import Path
class VectorDatabase:
    def __init__(self, dimension=1536):
        self.dimension = dimension
        # self.index = faiss.IndexFlatL2(dimension)
        self.index = faiss.IndexFlatIP(dimension)  # 使
        self.texts = []
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.max_length = 50

    def reset(self):
        self.index = faiss.IndexFlatIP(self.dimension)

    def save_faiss(self, path):
        # full_path = Path(directory) / file_name_faiss
        # try:
        index_bytes = faiss.serialize_index(self.index)
        with open(path, 'wb') as f:
            f.write(index_bytes)
        # except:
        #     faiss.write_index(self.index, path)

    def load_faiss(self, path):
        # try:
        with open(path, 'rb') as f:
            index_bytes = f.read()
        index_array = np.frombuffer(index_bytes, dtype='uint8')
        self.index = faiss.deserialize_index(index_array )
        # except:
        #     self.index = faiss.read_index(path)

    def save_texts(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.texts, f)
    def load_texts(self, path):
        with open(path, 'rb') as f:
            self.texts = pickle.load(f)

    def _get_embedding(self, text):
        embedding = None  # 初始化 embedding
        count = 0
        while count < 3:
            try:
                embedding = client.embeddings.create(input = [text], model='text-embedding-ada-002').data[0].embedding
                break

            except:
                time.sleep(3)
                count += 1
        if embedding is None:
            embedding = [random.uniform(-1, 1) for _ in range(1536)]
        embedding = np.array(embedding)
        embedding = embedding.reshape(1, -1)

        return embedding



    def merge_strings_until_max_length(self, strings, max_length, index):
        merged_list = []
        for i in range(len(strings)):

            if len(self.tokenization(''.join(merged_list))) + len(self.tokenization(strings[i])) > max_length:
                break
            merged_list.append(strings[i])


        index = index[:len(merged_list)]
        index_2 = [sorted(index).index(x) for x in index]
        original_order = sorted(enumerate(merged_list), key=lambda x: index_2[x[0]])
        restored_sentences = [sentence for i, sentence in original_order]
        return ''.join(restored_sentences),restored_sentences


    def tokenization(self, text):
        return self.encoding.encode(text)

    def add_text(self, text, split=True):

        if split:
            segments = self.split_text(text, self.max_length)
            for sentence in segments:
                embedding = self._get_embedding(sentence)
                if self.index.is_trained:
                    self.index.add(embedding)
                self.texts.append(sentence)
        else:
            embedding = self._get_embedding(text)
            if self.index.is_trained:
                self.index.add(embedding)
            self.texts.append(text)

    def query(self, query_text, max_length, return_list=False):

        query_embedding = self._get_embedding(query_text)
        D, I = self.index.search(query_embedding, 99999)

        relative_sentense =  [self.texts[i] for i in I[0] if i >= 0]
        index = [i for i in I[0] if i >= 0]
        text, text_list = self.merge_strings_until_max_length(strings=relative_sentense, max_length=max_length, index=index)
        if return_list:
            return text_list
        return text

    def query_num(self, query_text, num):

        query_embedding = self._get_embedding(query_text)
        D, I = self.index.search(query_embedding, 99999)  # 执行搜索
        # [(self.texts[i], D[0][j]) for j, i in enumerate(I[0])]
        relative_sentense =  [self.texts[i] for i in I[0] if i >= 0]
        if len(relative_sentense) < num:
            return ''.join(relative_sentense)
        relative_sentense = relative_sentense[:num]
        return ''.join(relative_sentense)


    def find_sentences(self, text):


        sentence_endings = r'([。？！.?!\n]+)'
        sentences = re.split(sentence_endings, text)

        sentences = [sentences[i] + sentences[i+1] for i in range(0, len(sentences)-1, 2)]
        sentences = [i.strip() for i in sentences]
        sentences = [i for i in sentences if i != '']
        return sentences

    def split_text(self, text, max_length):
        sentences = self.find_sentences(text)
        segments = []  # 存储最终的文本段
        current_segment = []  # 当前正在构建的段
        current_length = 0  # 当前段的长度

        for sentence in sentences:
            # 估计当前句子的token长度
            sentence_length = len(self.tokenization(sentence))
            if sentence_length > max_length:
                # 如果句子长度本身就超过最大长度，将其作为一个独立的段落
                if current_segment:
                    segments.append(''.join(current_segment))  # 先添加当前正在构建的段
                    current_segment = []  # 重置当前段
                    current_length = 0
                segments.append(sentence)  # 添加超长句子作为独立段落
            elif current_length + sentence_length <= max_length:
                # 如果当前句子可以加入当前段
                current_segment.append(sentence)
                current_length += sentence_length
            else:
                # 如果当前句子加入会超过最大长度，先结束当前段
                segments.append(''.join(current_segment))  # 将句子合并为一个段落
                current_segment = [sentence]  # 开始新的段
                current_length = sentence_length

        if current_segment:
            segments.append(''.join(current_segment))

        return segments

if __name__ == '__main__':

    multiline_string = """你姓“孙”，小名“甜儿”，生于清光绪十二年（1886年），是巢湖当地人，家住“荒山”北，父母去世得早，哥哥【孙咸恩】比你大10岁，去湖中画舫上做水夫赚钱养你，有时会把船上的好吃的装在一个食盒里，带回家给你。
    你8岁时（1893年），哥哥为了娶妻，到“崔庄”借钱，欠下一笔债·······哥哥婚后不久，“崔庄”的护院【周蒙当】到你家讨利息，见你哥还不出，就找来一位“张婆子”，把你带走，送到债主的家里干活，充抵利息。
    “张婆子”说你家根本还不起钱，不如把你卖的画舫上，多少能换回几两银子······多亏太太（汪氏）的丫鬟【宝柳】为你说话，才把你留下--太太根据你的小名，给你改名【宝柑】，做了丫鬟--你那时靠“宝柳姐”照顾，她教你如何小心做事，又安慰你说等你哥凑到钱，就能来接你······
    次年（1894年），太太的妹妹【昔颜】被接来“崔庄”住，老爷【崔寿亨】让你去贴身服侍【昔颜】，和她一起住“北院”（昔颜房）。
    又过了一年（1895年），你眼看着中秋已过，凉风渐起，仍不见哥哥来赎你，就愈发想家，伺候【昔颜】时心不在焉，给她端来的洗脸水里只有热水，烫了她一下，被她张口大骂--你满心委屈，哭了起来。
    【昔颜】抬手就要打你，你马上跑了出去--【昔颜】追到院子里，遇到“宝柳姐”，被她拦住，才没有打成，仍不肯罢休，对你说等你回来再打死你！你吓坏了，躲在“宝柳姐”身后，哭着求她帮帮你--你想回家！
    “宝柳姐”劝你打消这种念头，但你这次没有再听她的话，偷偷逃出“崔庄”，跑去“大码头”，一直等到画舫靠岸，看到哥哥从上面下来，才跑过去，说要跟他回家······哥哥看到你，先是大喜，然后皱起眉头，担心你跑了之后，债主不会善罢甘休，要去你家里抓人，便摇头说不能带你回家。
    为了不让你被抓，哥哥带你躲到“荒山”脚下，让你藏在一个亭子后面的山洞里，他回家用食盒给你装来几个窝头，又带来一件厚衣服，之后隔两天就给你送一次吃的······你本以为过一段时间就没事了，没想到【周蒙当】竟跟踪哥哥，找到洞口--哥哥想拦阻【周蒙当】进洞，被他一脚踹倒。
    哥哥要保护你，抄起地上的石头去打【周蒙当】，没想到【周蒙当】眼疾手快，举起随身带的弩，一箭射伤了哥哥的腿，疼得他倒地哀嚎！
    你跑到哥哥身边，抱住他，眼看着哥哥腿上流出的血染红了你的衣服--【周蒙当】大步走过来，你想打他，却被他像拿小鸡一样抓起来，带回“崔庄”，在交给【昔颜】前，先把你关进“南院”，直到你饿得没有力气，保证不再逃走后，才送你去“北院”，继续服侍【昔颜】。
    【崔寿亨】随后做主，把“宝柳姐”嫁给【周蒙当】-你听说后，跑去找“宝柳姐”，想告诉她【周蒙当】不是好人！“宝柳姐”不等你开口，就抱住了你，说出一个噩耗--原来你哥的腿被【周蒙当】用弩箭射伤后，没有及时求医，导致伤口恶化，高烧数日不退······在昨晚去世了！  
    几天后，【周蒙当】在“崔庄”里娶了“宝柳姐”。你牢记哥哥的仇，眼睁睁看着“宝柳姐”嫁给仇人后，就迁怒她，不再去找她，只听说她把弟弟也接来“崔庄”，此外还有【周蒙当】的弟弟【周持同】和他们住在一起--你被关在“南院”时，【周持同】曾给你送过水-一那时“南院”里的水井已是一口枯井，【周蒙当】用铁链把一块木板锁在上面，封住井口。
    第二年（1896年），“宝柳姐”生下一个男孩，因为不识字，就去请太太给儿子取名【琏逸】（周琏逸）--太太那时也有了身孕，“宝柳姐”坐完月子后抱着【琏逸】搬回“北院”，住在“西屋”，一边带孩子，一边伺候太太。
    【昔颜】每天都带你去“家主房”看太太，你因此和“宝柳姐”见了面，看着她为仇人养孩子，就更加气愤，有意躲开她--你那段时间没有见到【周蒙当】，急于知道仇人去向，又不愿找“宝柳姐”问，最后只能装作去找【周持同】道谢（为他当时给你送水），从他的口中得知【周蒙当】在“宝柳姐”怀孕时，外出投军······你此后常去找【周持同】聊天，借机探听他哥的情况，渐渐熟络--在【周持同】面前，你仍用“宝柳姐”来称呼【宝柳】。
    初冬，太太生下女儿，取名【嫣楠】（崔嫣楠）。
    之后几年，【周蒙当】每年都回“崔庄”探亲，你却找不到机会报仇。
    你14岁时（1899年），听说太太想给【昔颜】找个婆家，却没定下来。
    一年后（1900年），你去【周持同】的住处找他时，见屋里没有人，墙上挂着一把“大钥匙”，从尺寸上能看出是当年锁枯井上木板的······你摘下钥匙，去“南院”打开枯井上的木板，再把钥匙放回去，并未被人察觉--你始终忘不了哥哥的仇，终于想通一个早该明白的真相--当年你逃走前，只跟“宝柳姐”说过想要回家，一定是她告的密，她还嫁给仇人，简直该死！
    第二天晚上，你和【周持同】聊天时，看“宝柳姐”抱着【琏逸】去【周持同】住的屋子，就决心动手！你回到“北院”，等“宝柳姐”抱着孩子回“西屋”后，就在院子里把她从屋里叫出来--你故意没叫“宝柳姐”，只叫【宝柳】，等她出屋后，骗她跟着你去“南院”，从背后举起石块打击她的后脑，再把尸体推下枯井，锁上木板······“宝柳姐”失踪后，太太让【周持同】把【琏逸】带出有女眷居住的“北院”，与他和“宝柳姐”的弟弟一起住。
    你后来听【周持同】说他哥回来探亲没见到妻子，去庄外的河边喝闷酒，喝醉后回来说等他当上将官，就接亲人离开“崔庄”······
    中秋后，太太身体不舒服，把你借来“家主房”帮她照顾【嫣楠】-一太太从不打骂你，还教你识字，你因此用心照看【嫣楠】，不久听说【昔颜】在她房里要死要活······秋末，太太的病刚见好，就去【昔颜】那里陪她住。
    入冬后，【昔颜】嫁给【崔寿亨】做妾，和她姐姐共事一夫······【崔寿亨】很宠【昔颜】，见她与太太争吵，说不愿和姐姐住在一起，就同意她留在原来的屋里，他去找她过夜--“家主房”和“昔颜房”门对门，到了晚上，年幼的【嫣楠】不见【崔寿亨】回“家主房”，吵着要去小姨那里叫爹爹回来--太太只能叹气，加上有意回避，就让你抱着【嫣楠】，和她去“南院”里过夜······【嫣楠】仍哭着要爹爹，你怕她靠近枯井，就紧紧抱住她。  
    你16岁时（1901年），被太太留下，【崔寿亨】打算给【昔颜】买一个新丫鬟，去“家主房”拿银子时才发现床下的上锁箱子里少了钱，不禁大怒。
    【崔寿亨】首先想到是家贼，去“前院”翻找了所有人的住处，并没有找到那笔银子，就怀疑太太偷拿了······太太气不过，当晚就病倒了，郎中诊断是受了寒-【崔寿亨】听太太咳嗽，担心传染，竟让太太搬去“西屋”住--你知道【崔寿亨】怕受寒生病，每年过了中秋，就会在屋里放一个铁炭盆，晚上放入烧着的炭取暖，为避免中“炭毒”，他烧炭时会打开房内的小窗透气。
    【嫣楠】不愿离开母亲，你就带她到“西屋”住，同时照顾太太，每天煎药······即便如此，太太也没熬过冬天。
    太太去世那天，你听说【昔颜】让“厨房”里做了“赤豆酒酿”，就跟太太聊起这事，太太说她也想喝·······你去“厨房”给太太拿时，听说做好的“赤豆酒酿”都被老爷端走了，便去“昔颜房”里要。【崔寿亨】给了你一碗，让你加些冰糖再熬煮一下·······你去“厨房”加了冰糖熬煮，端回来给太太。
    太太喝完后没多久，病情突然恶化，开始咳血，说要是她死了，就把首饰都留给【昔颜】······之后等不到郎中来，太太就咽了气，尸体脸色乌青。
    太太去世后，【嫣楠】哭得非常伤心，你一直陪在她身边······后来你和【周持同】聊天时，说了太太的事，听他说这一年他哥都没有回“崔庄”。
    转过年后（1902年），【昔颜】被扶正，成了新的太太，带着【崔寿亨】给她新买来的丫鬟（宝桂），仍留在原先的住处（昔颜房）。
    夏末，【周蒙当】返回“崔庄”，要接走他的弟弟和儿子，还有“宝柳姐”的弟弟-你直到【周持同】来找你道别时，才知道仇人不会再回“崔庄”，只能假意对【周持同】不舍，让他答应保持联络，以免从此失去仇人踪迹。
    三年后（1904年），你被【崔寿亨】收做姨娘，让你从“西屋”搬到“家主房”，说要给你改个不像丫鬟的名字。你想起小名和哥哥的名字，要牢记他当年被害之仇，就改名【甜愁】·······【嫣楠】舍不得你，为此跑到她父亲面前吵闹。【崔寿亨】非常生气，不但骂了没了娘的【嫣楠】，还让她搬出“北院”，到“南院”里住，改由一个大丫鬟（宝杨）照顾。
    你虽然怕去“南院”，但还会时常到那里看望【嫣楠】，有时会在那里见到【昔颜】--你不敢留在“南院”过夜，好在自己是姨娘，晚上要回“家主房”陪老爷·······父亲自从丢了银子，就格外小心，不但随身带着锁箱子的钥匙，睡觉前也一定会从“家主房”里面插上房门，你回来晚了都进不去。
    民国元年（1912年），男人剪去辫子，不再严禁男女见面。
    去年（1913年），你听【嫣楠】说想上学，就去【崔寿亨】面前求情，让她去合肥城里的新式学堂住校读书--此后【嫣楠】只在逢节或放假时才回来，原本照顾她的大丫鬟搬到“北院”的“西屋”住，继续伺候女眷。
    今年（1914年），9月，【嫣楠】从学堂回来后，一直留在“崔庄”，全家一起过了中秋（10月4日）。
    10月12日，你听说【周持同】来“崔庄”拜访，因为他以前住的屋子已经被当成“杂物房”，【崔寿亨】就留【周持同】住在“前院”的“客房”。  
    你要打听仇人动向，就偷偷去“客房”和【周持同】见面，聊起往事，说自己迫于无奈做了小妾、改名等事······【周持同】突然激动起来，说他这些年都没有忘记你，随后又说他要让【崔寿亨】给他投资建厂，过几天大哥就来和【崔寿亨】面谈······事成之后，他就是厂长！
    你听说【周蒙当】要和【崔寿亨】见面，想到终于有机会动手报仇，忍不住笑了出来--你怕露出破绽，赶紧望着窗外站起身，说要回“家主房”服侍老爷--你离开时，感觉【周持同】对你有不舍之意。
    在接下来的两天里，你每天都去“客房”找【周持同】相处片刻，见他一副心急火燎的模样，只盼与你共度良宵，就知道将来能利用他。
    10月15日，下午，【崔寿亨】说他要码头赴一位故人之约，然后去“客房”找【周持同】，一同乘车出门--你猜故人就是【周蒙当】。
    【嫣楠】来“家主房”，拉上你和她一起去找【昔颜】聊天，期间听女佣说码头那边来了一艘大画舫，看到老爷去了上面······你又想到【崔寿亨】去和【周蒙当】见面的事，就说想去看热闹，【嫣楠】便央求身为太太的【昔颜】带你们出门。
    【昔颜】点头同意，你们三人梳妆后结伴外出，沿路绕过“崔庄”，走向湖边，看到大码头旁停着一艘三层高的画舫，从一层甲板上拉出两根长电线，一直接到大码头西侧的树上，连着数盏玻璃制品。
    一身西装的【周持同】和两个年轻男子站在树下，他们看到你们，就迎了过来。"""

    # # 示例使用
    db = VectorDatabase()
    db.add_text(multiline_string)
    # # 查询
    query_text = "你的哥哥是谁"
    results = db.query(query_text, 500)
    print("最相关的文本及其相似度分数：", results)
