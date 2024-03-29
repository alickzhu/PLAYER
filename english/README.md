# PLAYER
## English Dataset Description

This directory's subfolders contain datasets used for gaming and assessment purposes.
The JSON files contain the game scripts, while the CSV files contain the assessment questions.

### Script
Within the JSON folder, there is a file named script.json, which includes information about the game:
```
{
    "agent_num": "number of agents",
    "script_name": "script name",
    "character_name": ["list of agent names"]
}
```
Other JSON files contain detailed contents of the script, for example:
```
{
    "script": ["script content"],
    "acts_goal": ["goals"],
    "victims": [
    "victim 1",
    "victim 2",
    "victim 3",
    "victim 4"
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
Here, 
- `victims` list enumerates the victims in this script, which may include multiple individuals.
- `kill_by_me` corresponds to the list of victims; if a victim was killed by this character, then it is marked as 1, otherwise 0.
- `is_murderer` indicates whether the character has committed murder (regardless of the number of victims).

### Evaluation
The final_result folder contains the assessment questions we annotated.
The FSA.csv file is used to evaluate the Full Script Access method, summarizing the questions from other files.
The other CSV files are questions assessing each script.
- The `value` column refers to the type of question (A, B, C).
- The `type` column indicates whether the question is multiple-choice or single-choice (A for single-choice, B for multiple-choice).
- The `question` column is the question itself.
- The `a, b, c, d, e` columns are the options for the question. If a question does not have that many choices, the subsequent options will be null values.
- The `truth` column is the correct answer.