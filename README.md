# PLAYER*

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


