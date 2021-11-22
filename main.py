from DecisionTree import DecisionTreeNode
import Utilities

import pandas as pd
import numpy as np
import argparse
import copy

ap = argparse.ArgumentParser()
ap.add_argument("-D", "--data", required=True, help="File path to a .data file")
ap.add_argument("-H", "--header", required=True, help="File path to a .header file, corresponding to the .data file")
ap.add_argument("-t", "--target", required=True, help="Target attribute name appearing in the .header file")
ap.add_argument("-f", "--fraction-split", required=True, type=float,
                help="Fraction split into training (f) and test data (1-f)")
args = vars(ap.parse_args())


def id3(dataset, attributes_dict, target):
    root = DecisionTreeNode()

    most_common_val = None
    max_count = 0
    for value, count in dataset[target].value_counts().items():
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
    try:
        train, test, attribute_dict = Utilities.load_dataset(args["data"], args["header"], args["target"],
                                                             train_frac=args["fraction_split"])
        id3(train, attribute_dict, args["target"])
    except ValueError as ve:
        print(ve)
        print("Exiting...")
        exit(1)
