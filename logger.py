import os
import json
from collections import defaultdict


class ScoreLogger(object):
    def __init__(self, persistance_file_name="score.json", load_persistance_file=False):
        print("Create ScoreLogger with persistance file: ", persistance_file_name)
        self.persistance_file_name = persistance_file_name
        self.scores = defaultdict(dict)

        # try to load persistance file, if there are data, assign to `self.scores`
        if load_persistance_file and os.path.exists(persistance_file_name):
            with open(persistance_file_name, "r") as in_file:
                temp_scores = json.load(in_file)
                print(json.dumps(temp_scores, indent=2))
                if len(temp_scores.keys()) > 0:
                    self.scores = temp_scores

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
        if check_empty and len(self.scores.keys()) == 0:
            return
        if out_name is None:
            out_name = self.persistance_file_name
        with open(out_name, "w") as out_file:
            json.dump(self.scores, out_file, indent=2)

    def load(self, in_name=None):
        if in_name is None:
            in_name = self.persistance_file_name
        with open(in_name, "r") as in_file:
            self.scores = json.load(in_file)

    def print(self):
        print(json.dumps(self.scores, indent=2))


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
