import json
def json_format(character_data):

    character_data = character_data.strip().replace('\n','')
    try:
        character_data = eval(character_data)
    except:
        character_data = repair_json(character_data)
        if character_data is None:
            return {}
    return character_data

def repair_json(extract):
    if "{" in extract:
        index_position = [i for i, c in enumerate(extract) if c == "{"]
        index_position.reverse()
    else:
        return None
    for pos in index_position:
        data = extract[pos:]
        for i in range(len(data), 0, -1):
            try:
                # Try to parse the JSON
                character_data = data[:i]
                if '}' not in data:
                    character_data = character_data + '}'
                parsed = json.loads(character_data)
                # If successful, return the parsed JSON
                return parsed
            except json.JSONDecodeError:
                # If there's a JSON decode error, try with a shorter string
                continue
        # If no valid JSON is formed, return None or an empty dictionary
    return None
