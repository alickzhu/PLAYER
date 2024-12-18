# Questum

## Requirements

* Python >= 3.10
* pandas >= 1.5.3
* faiss-cpu >= 1.7.4
* transformers >= 4.38.2

## Dataset
The bilingual dataset is housed within the `./chinese` and `./english` folders. For detailed information, please refer to the `README.me files` located in these folders, available in both **Chinese** and **English** versions, respectively.


## Gameplay and Evaluation
To run our method on the Chinese dataset, please use the following:
```
Python main.py --script_name "孤舟萤（6人）" --output_root_path "./log_cn"
```

Here, `script_name` is the name of the script you want to run.

To run our method on the English dataset, please use the following:
```
Python main.py --script_name "Solitary Boat Firefly (6 people)" --output_root_path "./log_en" --is_english 1
```


The default model is the `GPT-3.5 16k` version. However, we also offer code support for using models such as LLaMA2 (7B, 13B, 70B) and Gemma7B. Please include the `--model_type` parameter and add the path to your model in the main.py file.


## Gameplay and Evaluation log
The log files are stored in the `./log_cn` and `./log_en` folders. The log files contain the following information:
* `*.xlsx`: The log file contains the dialogue history and the generated responses.
* `evaluation.xlsx`: The log file contains the evaluation results of the generated responses.


```
