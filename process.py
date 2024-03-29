dir_list = [
    "003-死穿白（8人开放）",
    "005-幽灵复仇（7人)",
    "033-丹水山庄（7人）",
    "053-未完结的爱（7人开放）",
    "054-东方之星号游轮事件（5人开放）",
    "131-罪恶（4人封闭）",
    "134-致命喷泉（4人封闭）",
    "141-校园不思议事件（5人）",
    "152-绝命阳光号（4人封闭）",
    "191-江湖客栈（4人封闭）",
    "1702-孤舟萤（6人）",
    "1849-曼娜（6人）",
]

dir_list = [
    "003-Death Wears White (open to 8 people)",
    "005-Ghost Revenge (7 people)",
    "033-Danshui Villa (7 people)",
    "053-Unfinished Love (open to 7 people)",
    "054-Oriental Star Cruise Incident (open to 5 people)",
    "131-Sin (4 people closed)",
    "134-Deadly Fountain (4 people closed)",
    "141-Unbelievable Incident (5 people)",
    "152-Desperate Sunshine (4 people closed)",
    "191-Riverside Inn (4 people closed)",
    "1702-Solitary Boat Firefly (6 people)",
    "1849-Manna (6 people)",
    "003-死穿白（8人开放）",
    "005-幽灵复仇（7人)",
    "033-丹水山庄（7人）",
    "053-未完结的爱（7人开放）",
    "054-东方之星号游轮事件（5人开放）",
    "131-罪恶（4人封闭）",
    "134-致命喷泉（4人封闭）",
    "141-校园不思议事件（5人）",
    "152-绝命阳光号（4人封闭）",
    "191-江湖客栈（4人封闭）",
    "1702-孤舟萤（6人）",
    "1849-曼娜（6人）",
]


import os

# 给定的根目录

# 保存着一些文件夹名字的列表
import os

root_dir = './chinese'
# 保存着一些文件夹名字的列表
folders_to_rename = dir_list

# 遍历根目录及其所有子目录
for dirpath, dirnames, filenames in os.walk(root_dir):
    for dirname in dirnames:
        # 完整的文件夹路径
        folder_path = os.path.join(dirpath, dirname)
        # 如果文件夹名在我们的列表中
        if dirname in folders_to_rename:
            # 使用"-"做分割，取第二个部分作为新名字
            new_name = dirname.split("-")[1] if "-" in dirname else dirname
            new_folder_path = os.path.join(dirpath, new_name)
            # 重命名文件夹
            os.rename(folder_path, new_folder_path)
            print(f"文件夹 {folder_path} 已重命名为 {new_folder_path}")

# 注意：执行完这段代码后，folders_to_rename中的文件夹名将不再有效，因为它们已经被重命名了。
