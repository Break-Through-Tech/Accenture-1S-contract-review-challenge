import json
from sklearn.model_selection import train_test_split

with open("../data/train_separate_questions.json", "r", encoding="utf-8") as f:
    data = json.load(f)

train_data, val_data = train_test_split(data['data'], test_size=0.20, random_state=42)

val_dict = dict()
train_dict = dict()
val_dict['version'] = data['version']
train_dict['version'] = data['version']
val_dict['data'] = val_data
train_dict['data'] = train_data

with open("../data/train.json", "w", encoding="utf-8") as f:
    json.dump(train_dict, f, indent=4)

with open("../data/val.json", "w", encoding="utf-8") as f:
    json.dump(val_dict, f, indent=4)