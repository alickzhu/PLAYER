import os
import json
import os
def process_json_files(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if 'all.json' in file:
                    os.remove(file_path)
                    continue
                if 'qa-cha' in file:
                    os.remove(file_path)
                    continue
                if 'relation' in file:
                    os.remove(file_path)
                    continue
                # 对 script_info.json 文件进行特殊处理
                if file == 'script_info.json':
                    keys_to_keep = ['agent_num', 'script_name', 'character_name']
                else:
                    keys_to_keep = ['script', 'acts_goal', 'victims', 'kill_by_me', 'is_murderer']

                # 删除不需要的键
                data = {k: data[k] for k in keys_to_keep if k in data}

                # 写入修改后的内容
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                    f.close()
def process_json_ficsv(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)

                if 'relation' in file:
                    os.remove(file_path)



process_json_ficsv('./chinese')