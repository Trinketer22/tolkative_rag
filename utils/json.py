import json
from typing import Optional, Callable


def load_json_dump(path: str, obj_constuctor: Optional[Callable] = None):
    parsed_data = []
    with open(path, "r", encoding="utf8") as data_file:
        for line in data_file:
            parsed_obj = json.loads(line)
            new_obj = (
                parsed_obj if obj_constuctor is None else obj_constuctor(parsed_obj)
            )
            parsed_data.append(new_obj)
    return parsed_data


def save_json_dict(data, path: str):
    with open(path, "w", encoding="utf8") as dump_file:
        dump_file.write(json.dumps(data))


def load_json_dict(path: str):
    with open(path, "r", encoding="utf8") as dump_file:
        return json.loads(dump_file.read())
