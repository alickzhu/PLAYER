# import os
# import shutil
#
# # 定义起始路径
# base_path = './'
#
# import json
#
# # 定义写入JSON的数据
# json_data = {
#     "acts_num": 3,
#     "script": [],
#     "acts_goal": [],
#     "questions": [],
#     "victims": [],
#     "final_goal": "",
#     "acts_performance": [],
#     "secrets": "",
#     "is_murderer": 1
# }
#
# # 遍历chinese文件夹下的所有子文件夹
# for root, dirs, files in os.walk(base_path):
#     # 确定当前目录是purpose文件夹
#     if os.path.basename(root) == 'purpose':
#         parent_dir = os.path.dirname(root)  # 获取上级目录路径
#         if '1702' in parent_dir:
#             continue# 获取上级目录路径
#
#
#     # 定位json和final_result文件夹
#         json_path = os.path.join(parent_dir, 'json')
#         final_result_path = os.path.join(parent_dir, 'final_result')
#
#         # 确保文件夹存在
#         if not os.path.exists(json_path):
#             os.makedirs(json_path)
#         if not os.path.exists(final_result_path):
#             os.makedirs(final_result_path)
#
#         # 遍历purpose文件夹中的txt文件
#         for file in files:
#             if file.endswith('.txt'):
#                 # 构建新的文件名
#                 json_filename = os.path.splitext(file)[0] + '.json'
#                 csv_filename = os.path.splitext(file)[0] + '.csv'
#
#                 # 格式化写入.json文件
#                 with open(os.path.join(json_path, json_filename), 'w') as f_json:
#                     json.dump(json_data, f_json, indent=4)
#                 # 创建一个空的.csv文件
#                 open(os.path.join(final_result_path, csv_filename), 'w').close()
#
# # 操作完成
# "JSON和CSV文件已按要求更新。"


# 由于代码执行环境重置，需要重新导入模块和定义变量
import os
import json

# # 重新定义基本路径和数据模板
# base_path = './'
# script_info_template = {
#     "agent_num": 5,
#     "script_name": "",
#     "character_name": [],
#     "acts_num": 3,
#     "stage": ["", "", "", "", "", "", ""],
#     "open_discuss_rounds": [2, 2, 2],
#     "keep_asking": [2, 2, 2],
#     "privacy_discuss_rounds": [0, 0, 1]
# }
#
# # 重新遍历chinese文件夹下的所有子文件夹
# for root, dirs, files in os.walk(base_path):
#     if 'json' in dirs:  # 确保json文件夹存在
#         json_path = os.path.join(root, 'json')
#         script_info_data = script_info_template.copy()  # 复制数据模板
#
#         # 获取script_name
#         script_name_parts = os.path.basename(root).split('-')[1].split('（')
#         script_info_data['script_name'] = script_name_parts[0].strip()
#
#         # 获取character_name
#         character_names = [os.path.splitext(file)[0] for file in os.listdir(json_path) if file.endswith('.json') and file != 'script_info.json']
#         script_info_data['character_name'] = character_names
#         script_info_data['agent_num'] = len(character_names)
#
#         # 写入script_info.json
#         script_info_path = os.path.join(json_path, 'script_info.json')
#         with open(script_info_path, 'w', encoding='utf-8') as f_script_info:
#             json.dump(script_info_data, f_script_info, ensure_ascii=False, indent=4)
#
# "script_info.json文件已在每个json文件夹中创建并写入指定内容。"




import os
import json

def create_qa_character_json(base_dir):
    # 遍历base_dir目录下的每个子目录
    for subdir, dirs, files in os.walk(base_dir):
        # 检查子目录中是否有名为'json'的目录
        if 'json' in dirs:
            json_dir_path = os.path.join(subdir, 'json')
            json_files = [f for f in os.listdir(json_dir_path) if f.endswith('.json') and 'script_info' not in f]
            # 创建一个字典，其中键是文件名（不含.json扩展名），值是["","",""]
            qa_dict = {os.path.splitext(f)[0]: ["", "", ""] for f in json_files}
            # 创建qa-character.json文件，并写入数据
            qa_file_path = os.path.join(subdir, 'qa-character.json')
            # if not os.path.exists(qa_file_path):
            with open(qa_file_path, 'w', encoding='utf-8') as qa_file:
                json.dump(qa_dict, qa_file, ensure_ascii=False, indent=4)

# 假定基础目录是'/mnt/data/chinese'，这是一个示例路径
base_dir = './'
create_qa_character_json(base_dir)
#
# # 验证一下操作是否成功，我们列出一些子目录和qa-character.json文件的内容
# # 为了验证，我将从一个子目录读取qa-character.json的内容并展示
# example_subdir = next(os.walk(base_dir))[1][0]  # 取第一个子目录
# example_qa_path = os.path.join(base_dir, example_subdir, 'qa-character.json')
# if os.path.exists(example_qa_path):
#     with open(example_qa_path, 'r', encoding='utf-8') as example_qa_file:
#         example_content = json.load(example_qa_file)
# else:
#     example_content = "qa-character.json 文件尚未创建或找不到。"
#
# example_content
