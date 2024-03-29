# PLAYER

中文数据集说明

该目录的子文件夹包含了用于游戏和评估的数据集。
其中json文件包含了游戏的剧本，而csv文件包含了评估的问题。

### 剧本
在JSON文件夹中有一个名为`script.json`的文件，其中包含了游戏的信息：
```json
{
    "agent_num": "agent的数量",
    "script_name": "剧本名字",
    "character_name": ["agent的名字"列表]
    ]
}
```
其他的json文件包含了剧本的具体内容，例如:
```json
{
    "script": ["剧本内容"],
    "acts_goal": ["目标"],
    "victims": [
        "受害者1",
        "受害者2",
        "受害者3",
        "受害者4"
    ],
    "kill_by_me": [
        0,
        0,
        1,
        1
    ],
    "is_murderer": 1
}
```
其中victims列举出了这个剧本中的受害者列表，可能包含多个人。
"kill_by_me"与受害者列表一一对应，如果该受害者是这个角色杀的，那么为1，否则为0。
"is_murderer"为该角色是否杀了人。（无论杀了几个人）


### Evaluation
final_result文件夹中包含了我们标注的测评问题。
其中`FSA.csv`是用来评测`Full Script Access方法`的问题，是其他文件问题的汇总。
其他的`csv`文件是测评每个剧本的问题。
其中value列指的是问题的类型（A,B,C)
type指的是问题是多选题还是单选题（A为单选，B为多选）
question是问题本身
a,b,c,d,e为问题的选项。如果一个问题没有那么多选择，则后面的为空值
truth为标准答案