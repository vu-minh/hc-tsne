import os
import json
from datetime import datetime
from collections import defaultdict


def now():
    return f"{datetime.now():%Y-%m-%d %H:%M:%S%z}"


class ScoreLogger(object):
    def __init__(self, persistance_file_name=None):
        self.persistance_file_name = persistance_file_name
        print("[DEBUG] ScoreLogger @", persistance_file_name)

        self.scores = defaultdict(dict)
        self.scores["meta"]["create_time"] = now()

    def reset(self):
        self.scores = defaultdict(dict)

    def log(self, key, value, method="default"):
        self.scores[method][key] = value

    def logs(self, list_key_value, method="default"):
        for (key, value) in list_key_value:
            self.scores[method][key] = value

    def log_dict(self, d, method="default"):
        self.scores[method].update(d)

    def score_names(self, method="default"):
        return list(self.scores[method].keys())

    def get_score(self, key, method="default"):
        return self.scores[method][key]

    def dump(self, out_name=None, check_empty=True):
        if check_empty and len(self.scores.keys()) <= 1:  # meta key
            return

        if out_name is None:
            out_name = self.persistance_file_name

        self.scores["meta"]["finish_time"] = now()
        with open(out_name, "w") as out_file:
            json.dump(self.scores, out_file, indent=2)

    def load(self, in_name):
        with open(in_name, "r") as in_file:
            self.scores = json.load(in_file)

    def print(self):
        print("\n[DEBUG] ScoreLogger @", self.persistance_file_name)
        temp_scores = {
            k0: {k1: v1 for k1, v1 in v0.items() if not isinstance(v1, list)}
            for k0, v0 in self.scores.items()
        }
        print(json.dumps(temp_scores, indent=2))


class LossLogger(object):
    def __init__(self, persistance_file_name="loss.json"):
        self.persistance_file_name = persistance_file_name
        self.loss = defaultdict(list)

    def reset(self):
        self.loss = defaultdict(list)

    def log(self, key, value):
        self.loss[key].append(value)

    def loss_names(self):
        return list(self.loss.keys())

    def get_loss(self, key):
        return self.loss[key]

    def dump(self, out_name=None, check_empty=True):
        if check_empty and len(self.loss.keys()) == 0:
            return
        if out_name is None:
            out_name = self.persistance_file_name
        with open(out_name, "w") as out_file:
            json.dump(self.loss, out_file, indent=2)

    def load(self, in_name=None):
        if in_name is None:
            in_name = self.persistance_file_name
        with open(in_name, "r") as in_file:
            self.loss = json.load(in_file)


if __name__ == "__main__":
    logger = LossLogger()

    for i in range(30):
        logger and logger.log("a", i)
        if i % 5 == 0:
            logger and logger.log("a5", i)

    print(logger.loss_names())
    print(logger.get_loss("a"), logger.get_loss("a5"))

    logger.reset()

    for i in range(5):
        logger.log("x", i * 100)
    print(logger.get_loss("x"))

    score_logger = ScoreLogger()
    score_logger.log("s", 100)
    score_logger.logs([("t", 1), ("b", 2)])
    score_logger.log_dict({"m": 9, "n": 999, "p": 99999})
    score_logger.print()
