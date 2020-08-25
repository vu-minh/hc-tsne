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


if __name__ == "__main__":
    logger = LossLogger()

    for i in range(30):
        logger.log("a", i)
        if i % 5 == 0:
            logger.log("a5", i)

    print(logger.loss_names())
    print(logger.get_loss("a"), logger.get_loss("a5"))

    logger.reset()

    for i in range(5):
        logger.log("x", i * 100)
    print(logger.get_loss("x"))
