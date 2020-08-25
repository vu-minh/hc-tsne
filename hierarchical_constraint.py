# Data structure to hold the hierarchical constraints in form of tree.

import numpy as np
from collections import namedtuple
from anytree import NodeMixin, RenderTree, LevelOrderIter, LevelGroupOrderIter


GroupConstraint = namedtuple("GroupConstraint", ["name", "alpha", "list_groups"])


class Group(object):
    """Group object, stores `items` of the same logical group/class."""

    def __init__(self, items=[]):
        self.level = -1  # level of node
        self.items = items
        self.update_centroid()

    def update_centroid(self, embedding=None):
        if len(self.items) == 0 or embedding is None:
            self.centroid = np.zeros((1, 2), dtype=np.float32)
        else:
            self.centroid = np.mean(embedding[self.items], axis=0, keepdims=True)


class GroupNode(Group, NodeMixin):
    """GroupNode is a `Group`, with `name` and `parent`, and is used in `anytree`"""

    def __init__(self, name, items=[], parent=None, children=None):
        super(GroupNode, self).__init__(items)

        self.name = name
        self.parent = parent
        if children:
            self.children = children


def show_tree(tree):
    for pre, fill, node in RenderTree(tree):
        # print(pre, fill, node)
        tree_str = f"{pre}{node.name}"
        print(tree_str.ljust(30), node.level, len(node.items))


def show_iterating_tree(tree, iterator=LevelOrderIter):
    # test iterating tree
    print("Iterate tree in level-ordre:")
    for node in iterator(tree):
        if not node.parent:
            p = "No parent"
        else:
            p = node.parent.name
        print(node.name.ljust(20), " <------ ", p)


def _update_level(tree):
    for level, children in enumerate(reversed(list(LevelGroupOrderIter(tree)))):
        for node in children:
            node.level = level + 1


class HierarchicalConstraint:
    """Util class to create hierarchical constraints from class labels"""

    def __init__(self, labels, label_names, label_percent=1.0):
        # store leaf nodes by node name
        self.G = dict()
        # store indices of points in each node by node name
        self.elements = dict()

        self.indices = np.arange(len(labels))  # indices of all data points
        self._create_leaf_nodes(labels, label_names, label_percent)

    def _create_leaf_nodes(self, labels, label_names, label_percent):
        for i, name in enumerate(label_names):
            if label_percent == 1.0:
                idx = self.indices[labels == i].tolist()
            else:
                n_select = int(label_percent * len(self.indices[labels == i]))
                idx = np.random.choice(
                    self.indices[labels == i], n_select, replace=False
                ).tolist()

            group_node = GroupNode(name, idx)
            self.G[name] = group_node
            self.elements[name] = idx

    def _create_intermediate_node(self, name, list_keys):
        elems = []  # list of indices of points
        children = []  # list of children nodes

        # merge all elements of each key in list_keys
        for key in list_keys:
            elems += self.elements[key]
            children.append(self.G[key])
        node = GroupNode(name, elems, children=children)

        # add back new created node to G for reusing
        self.G[name] = node
        self.elements[name] = elems
        return node


def _generate_constraints_fmnist(labels, label_names, depth=0, label_percent=1.0):
    H = HierarchicalConstraint(labels, label_names, label_percent)

    if depth == 2:
        H._create_intermediate_node("shoe", ["Sneaker", "Ankle boot"])
        H._create_intermediate_node("footwear", ["Sandal", "shoe"])
        H._create_intermediate_node("accessory", ["Bag"])
        H._create_intermediate_node("shirt", ["T-shirt/top", "Shirt"])
        H._create_intermediate_node("outerwear", ["Pullover", "Coat"])
        H._create_intermediate_node("long-shape", ["Dress", "Trouser"])
        H._create_intermediate_node("clothing", ["shirt", "outerwear", "long-shape"])
        root = ["footwear", "accessory", "clothing"]
    elif depth == 1:
        H._create_intermediate_node("footwear", ["Sandal", "Sneaker", "Ankle boot"])
        H._create_intermediate_node("accessory", ["Bag"])
        H._create_intermediate_node(
            "clothing", ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Shirt"]
        )
        root = ["footwear", "accessory", "clothing"]
    elif depth == 0:
        # flat groups: 10 classes under a root node
        root = label_names
    else:
        raise ValueError(f"Hierarchy with depth = {depth} is not supported!")

    # create root node
    tree = H._create_intermediate_node("root", root)
    return tree


def _generate_constraints_cifar10(labels, label_names, depth=0, label_percent=1.0):
    H = HierarchicalConstraint(labels, label_names, label_percent)

    # create intermediate groups
    if depth == 2:
        group_land_vehicles = H._create_intermediate_node(
            "land-vehicles", ["automobile", "truck"]
        )
        group_manmade = H._create_intermediate_node(
            "man-made", ["airplane", "ship", group_land_vehicles.name]
        )

        group_pets = H._create_intermediate_node("pets", ["dog", "cat"])
        group_hoofed_mammals = H._create_intermediate_node(
            "hoofed-mammals", ["deer", "horse"]
        )
        group_nature = H._create_intermediate_node(
            "nature", ["bird", "frog", group_pets.name, group_hoofed_mammals.name]
        )

        root = [group_manmade.name, group_nature.name]

    elif depth == 1:
        man_made = ["airplane", "automobile", "ship", "truck"]
        nature = ["bird", "cat", "deer", "dog", "frog", "horse"]

        group_manmade = H._create_intermediate_node("man-made", man_made)
        group_nature = H._create_intermediate_node("nature", nature)
        root = [group_manmade.name, group_nature.name]

    elif depth == 0:
        # flat groups: 10 classes under a root node
        root = label_names
    else:
        raise ValueError(f"Hierarchy with depth = {depth} is not supported!")

    # create root node
    tree = H._create_intermediate_node("root", root)
    return tree


def _generate_constraints_mnist(labels, label_names, depth=0, label_percent=1.0):

    H = HierarchicalConstraint(labels, label_names, label_percent)

    # create intermediate groups
    if depth == 2:
        g147 = H._create_intermediate_node("G-1-4-7", ["1", "4", "7"])
        g35 = H._create_intermediate_node("G-3-5", ["3", "5"])
        g0689 = H._create_intermediate_node("G-0-6-8-9", ["0", "6", "8", "9"])
        root = ["2", g147.name, g35.name, g0689.name]
    elif depth == 1:
        g147 = H._create_intermediate_node("G-1-4-7", ["1", "4", "7"])
        g235 = H._create_intermediate_node("G-2-3-5", ["2", "3", "5"])
        g0689 = H._create_intermediate_node("G-0-6-8-9", ["0", "6", "8", "9"])
        root = [g147.name, g235.name, g0689.name]
    elif depth == 0:
        # flat groups: 10 classes under a root node
        root = label_names
    else:
        raise ValueError(f"Hierarchy with depth = {depth} is not supported!")

    tree = H._create_intermediate_node("root", root)
    return tree


def _generate_constraints_flat(labels, label_names, config, depth=0, label_percent=1.0):

    H = HierarchicalConstraint(labels, label_names, label_percent)

    # create intermediate groups
    if depth == 2:
        ...
    elif depth == 0:
        # flat groups: 10 classes under a root node
        root = label_names
    else:
        raise ValueError(f"Hierarchy with depth = {depth} is not supported!")

    tree = H._create_intermediate_node("root", root)
    return tree


def generate_constraints(dataset_name, labels, label_names, depth=0, label_percent=1.0):
    generate_func = {
        "mnist": _generate_constraints_mnist,
        "fmnist": _generate_constraints_fmnist,
        "cifar10": _generate_constraints_cifar10,
    }[dataset_name]
    tree = generate_func(labels, label_names, depth=depth, label_percent=label_percent)

    # note to update the level of each node in the tree
    _update_level(tree)
    return tree
