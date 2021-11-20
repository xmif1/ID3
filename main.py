from DecisionTree import DecisionTreeNode

import pandas as pd
import numpy as np

import Utilities

data_file = ""
header_file = ""


# def id3(examples, attributes, target):
#     root = DecisionTreeNode(examples)

if __name__ == "__main__":
    train, test = Utilities.load_dataset(data_file, header_file)
