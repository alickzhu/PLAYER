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


## Murder Mystery Game Rules

### Rules

**Rule 1:** The total number of players participating in the game depends on the script. There may be one or more players who are the murderer(s), while the rest are civilians.  
**Rule 2:** The goal of the game is for civilian players to collaborate and face a meticulously planned murder case together, collecting evidence and reasoning to identify the real murderer among the suspects; murderer players must concoct lies to hide their identity and avoid detection, while also achieving other objectives in the game.  
**Rule 3:** Throughout the game, only murderer players are allowed to lie. To conceal their identity, murderers may choose to frame others to absolve themselves of guilt; non-murderer players (civilians) must answer questions from other players and the host honestly and provide as much information as they know about the case to help uncover the truth.  
**Rule 4:** The game host is only responsible for ensuring the game follows a specific process. They are not players in the game and do not participate in the storyline.  
**Rule 5:** At the start of the game, each player receives their personal character script from the host, which contains information about their role and identity.  
**Rule 6:** The game may have multiple acts, and your script will be updated accordingly.  
**Rule 7:** Other players cannot see the content of each player's personal character script, so players must and can only collect information about other players through interaction after the game starts.  
**Rule 8:** In the voting phase, each player needs to cast their vote for who they think is the murderer in each case (including themselves, although this is not encouraged). If the player with the most votes is the murderer, the civilian players win. Otherwise, the murderer players win.

### Gameplay

The game has one or more acts. At the beginning of the game, players introduce themselves according to the script, and in each act, you will receive more plot information.
In each act, you can ask questions, share your observations, or make deductions to help solve the murder case.
The goal is to identify the true murderer and explain their motive. If you are the true murderer, you must hide your identity and avoid detection.
