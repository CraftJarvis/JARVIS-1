import json
import pickle


def save_pickle(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_json(obj, filepath):
    with open(filepath, 'w') as f:
        json.dump(obj, f, indent=4)


def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)
