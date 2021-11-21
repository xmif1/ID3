from DecisionTree import DecisionTreeNode

import pandas as pd
import numpy as np

import Utilities
import copy

data_file = ""
header_file = ""


def id3(dataset, attributes_dict, target):
    root = DecisionTreeNode()

    most_common_val = None
    max_count = 0
    for value, count in dataset[target].value_counts():
        if max_count <= count:
            max_count = count
            most_common_val = value

        if count == dataset[target].shape[0]:
            root.classification = value
            return root

    if len(attributes_dict) == 0:
        root.classification = most_common_val
        return root

    root.split_attr = Utilities.best_attribute(dataset, target)
    for value in attributes_dict[root.split_attr]:
        subset = dataset.loc[dataset[root.split_attr] == value]

        if subset.shape[0] == 0:
            child = DecisionTreeNode()
            child.set_parent(root)
            child.node_attr = root.split_attr
            child.node_attr_val = value
            child.classification = most_common_val
        else:
            modified_attr_dict = copy.deepcopy(attributes_dict)
            modified_attr_dict.pop(root.split_attr)

            child = id3(subset, modified_attr_dict, target)
            child.set_parent(root)
            child.node_attr = root.split_attr
            child.node_attr_val = value

        root.children.append(child)

    return root


if __name__ == "__main__":
    train, test = Utilities.load_dataset(data_file, header_file)
