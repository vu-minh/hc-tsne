# create datasets using keras API

import numpy as np
from tensorflow import keras

dataset_objects = {
    "mnist": (keras.datasets.mnist, [f"{i}" for i in range(10)]),
    "fmnist": (
        keras.datasets.fashion_mnist,
        [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ],
    ),
    "cifar10": (
        keras.datasets.cifar10,
        [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ],
    ),
}


def load_dataset(dataset_name="mnist", N_train=10000, N_test=5000, debug=False):
    """Get keras dataset. ("mnist", "fmnist" and "cifar10").
    If `N_train` or `N_test` is set to None, all the train/test set will be returned.
    """
    dataset_obj, label_names = dataset_objects[dataset_name]
    (X_train, y_train), (X_test, y_test) = dataset_obj.load_data()

    np.random.seed(42)
    if N_train:
        idx = np.random.choice(X_train.shape[0], replace=False, size=N_train)
        X_train, y_train = X_train[idx], y_train[idx]
    if N_test:
        idx = np.random.choice(X_test.shape[0], replace=False, size=N_test)
        X_test, y_test = X_test[idx], y_test[idx]

    X_train = X_train.reshape(X_train.shape[0], -1).astype("float32") / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).astype("float32") / 255.0

    if debug:
        print(dataset_name, X_train.dtype, X_test.dtype)
        print(X_train.shape, y_train.shape, np.unique(y_train, return_counts=True))
        print(X_test.shape, y_test.shape, np.unique(y_test, return_counts=True))
        print(label_names)

    return (X_train, y_train), (X_test, y_test), label_names


if __name__ == "__main__":
    dataset_name = ["mnist", "fmnist", "cifar10"][0]
    (X_train, y_train), (X_test, y_test), label_names = load_dataset(
        dataset_name, debug=True
    )
