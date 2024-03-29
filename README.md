# PLAYER

This repository contains the code submitted to COLM for "PLAYER*: Enhancing LLM-based Multi-Agent Communication and Interaction in Murder Mystery Games."

The bilingual dataset is located in the [chinese](https://github.com/alickzhu/PLAYER/tree/main/chinese) and [english](https://github.com/alickzhu/PLAYER/tree/main/english) folders.

To run our method on the Chinese dataset, please use the following:
```
Python main.py --script_name "1702-孤舟萤（6人）" --output_root_path "./log_cn"
```

Here, `script_name` is the name of the script you want to run.

To run our method on the English dataset, please use the following:
```
Python main.py --script_name "1702-Solitary Boat Firefly (6 people)" --output_root_path "./log_en" --is_english 1
```