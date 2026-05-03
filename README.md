# PLAYER*: Enhancing LLM-based Multi-Agent Communication and Interaction in Murder Mystery Games

This repository contains the official code and dataset for the paper **[PLAYER*: Enhancing LLM-based Multi-Agent Communication and Interaction in Murder Mystery Games](https://arxiv.org/abs/2404.17662)**.

PLAYER* is a framework for studying multi-agent communication, interaction, and reasoning of LLM-based agents in **Murder Mystery Games (MMGs)**. We release **WellPlay**, a reasoning dataset for multi-agent conversational inference, and propose a sensor-based agent architecture that enables LLM agents to ask questions, gather evidence, infer hidden objectives, and reason about complex social relationships in adversarial multi-party settings.

**Keywords:** PLAYER*, WellPlay, murder mystery games, multi-agent LLM, agent communication, conversational inference, social reasoning, role-playing agents, hidden-role games, LLM evaluation.

- Paper: [arXiv:2404.17662](https://arxiv.org/abs/2404.17662)
- PDF: [2404.17662.pdf](https://arxiv.org/pdf/2404.17662)


## Requirements

* Python >= 3.10
* pandas >= 1.5.3
* faiss-cpu >= 1.7.4
* transformers >= 4.38.2

## Dataset

The bilingual **WellPlay** dataset is housed within the `./chinese` and `./english` folders. For detailed information, please refer to the `README.md` files located in these folders, available in both **Chinese** and **English** versions, respectively.


## Gameplay and Evaluation

To run our method on the Chinese dataset, please use the following:
```
python main.py --script_name "孤舟萤（6人）" --output_root_path "./log_cn"
```

Here, `script_name` is the name of the script you want to run.

To run our method on the English dataset, please use the following:
```
python main.py --script_name "Solitary Boat Firefly (6 people)" --output_root_path "./log_en" --is_english 1
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


## Related Work

This project is part of a broader line of work on narrative understanding, social reasoning, character relationship extraction, and interactive LLM-based systems.

### Conan: Understanding Complex Relationships in Detective Narratives

[Conan](https://aclanthology.org/2024.findings-acl.454/) is a **Findings of ACL 2024** paper that introduces a benchmark for narrative understanding, character relationship extraction, and relation graph analysis in detective narratives. It focuses on complex social relationships, role-oriented perspectives, public and secret character relations, and long-context reasoning for large language models, providing the foundation that PLAYER* builds upon for hidden-role multi-agent settings.

- Paper: [ACL Anthology](https://aclanthology.org/2024.findings-acl.454/)
- PDF: [2024.findings-acl.454.pdf](https://aclanthology.org/2024.findings-acl.454.pdf)
- Code: [BLPXSPG/Conan](https://github.com/BLPXSPG/Conan)
```bibtex
@inproceedings{zhao-2024-conan,
  title = {Large Language Models Fall Short: Understanding Complex Relationships in Detective Narratives},
  author = {Zhao, Runcong and Zhu, Qinglin and Xu, Hainiu and Li, Jiazheng and Zhou, Yuxiang and He, Yulan and Gui, Lin},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2024},
  pages = {7618--7638},
  year = {2024},
  address = {Bangkok, Thailand},
  publisher = {Association for Computational Linguistics},
  doi = {10.18653/v1/2024.findings-acl.454},
  url = {https://aclanthology.org/2024.findings-acl.454/}
}
```

### SymbolicThought: Integrating Language Models and Symbolic Reasoning for Consistent and Interpretable Human Relationship Understanding

[SymbolicThought](https://arxiv.org/abs/2507.04189) is an **ACL 2026 Demo** paper that further develops this line of work into a human-in-the-loop system for relationship understanding. It combines LLM-based extraction with symbolic reasoning, editable character relationship graphs, logical constraints, and interactive conflict resolution. In particular, it introduces **logical annotations between character relationships**, enabling the system to capture relation-level dependencies, consistency constraints, and conflicts among interpersonal relations for more interpretable social relation annotation.

- Paper: [arXiv:2507.04189](https://arxiv.org/abs/2507.04189)
- PDF: [2507.04189.pdf](https://arxiv.org/pdf/2507.04189)
- Code: [BLPXSPG/SymbolicThought](https://github.com/BLPXSPG/SymbolicThought)
```bibtex
@misc{zhao2025symbolicthought,
  title = {SymbolicThought: Integrating Language Models and Symbolic Reasoning for Consistent and Interpretable Human Relationship Understanding},
  author = {Zhao, Runcong and Zhu, Qinglin and Xu, Hainiu and Liang, Bin and Gui, Lin and He, Yulan},
  year = {2025},
  eprint = {2507.04189},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  doi = {10.48550/arXiv.2507.04189},
  url = {https://arxiv.org/abs/2507.04189},
  note = {ACL 2026 Demo}
}
```


## Citation

If you use PLAYER* or the WellPlay dataset in your research, please cite our paper:

```bibtex
@misc{zhu2024player,
  title = {{PLAYER}*: Enhancing LLM-based Multi-Agent Communication and Interaction in Murder Mystery Games},
  author = {Zhu, Qinglin and Zhao, Runcong and Liang, Bin and Du, Jinhua and Gui, Lin and He, Yulan},
  year = {2024},
  eprint = {2404.17662},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  doi = {10.48550/arXiv.2404.17662},
  url = {https://arxiv.org/abs/2404.17662}
}
```
