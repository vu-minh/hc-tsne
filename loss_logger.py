import json
from collections import defaultdict


class LossLogger(object):
    def __init__(self):
        self.loss = defaultdict(list)

    def reset(self):
        self.loss = defaultdict(list)

    def log(self, key, value):
        self.loss[key].append(value)

    def loss_names(self):
        return list(self.loss.keys())

    def get_loss(self, key):
        return self.loss[key]

    def dump(self, out_name, check_empty=True):
        if check_empty and len(self.loss.keys()) == 0:
            return
        with open(out_name, "w") as out_file:
            json.dump(self.loss, out_file)

    def load(self, in_name):
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
