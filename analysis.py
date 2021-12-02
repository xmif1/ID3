from matplotlib import pyplot as plt

from DecisionTree import DecisionTree
import Utilities

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-D", "--data", required=True, help="File path to a .data file")
ap.add_argument("-H", "--header", required=True, help="File path to a .header file, corresponding to the .data file")
ap.add_argument("-C", "--continuous", required=False, help="File path to a .continuous file, specifying the continuous "
                                                           "data attributes in the corresponding .header file")
ap.add_argument("-t", "--target", required=True, help="Target attribute name appearing in the .header file")
ap.add_argument("-m", "--missing", required=False, help="Missing attribute value flag")
args = vars(ap.parse_args())


if __name__ == "__main__":
    pre_pruning_test = []
    pre_pruning_train = []
    post_pruning_test = []
    post_pruning_train = []

    for f in range(1, 11):
        frac = f / 10.0
        try:
            train, test, attributes_dict = Utilities.load_dataset(args["data"], args["header"], args["continuous"],
                                                                  args["target"], args["missing"], frac)

            decisionTree = DecisionTree(train, attributes_dict, args["target"], args["missing"])
            train_non_missing = decisionTree.dataset.loc[decisionTree.dataset[args["target"]] != args["missing"]]

            for attr in train_non_missing.columns:
                if attr != args["target"]:
                    train_non_missing = train_non_missing.loc[train_non_missing[attr] != args["missing"]]

            pre_pruning_test.append([frac, decisionTree.benchmark(test)])
            pre_pruning_train.append([frac, decisionTree.benchmark(train_non_missing)])

            decisionTree.prune_tree(test)

            post_pruning_test.append([frac, decisionTree.benchmark(test)])
            post_pruning_train.append([frac, decisionTree.benchmark(train_non_missing)])
        except ValueError as ve:
            print(ve)
            print("Exiting...")
            exit(1)

    plt.title("Overfitting Analysis Pre and Post Pruning")

    plt.plot([x[0] for x in pre_pruning_test], [x[1] for x in pre_pruning_test], label="Pre-pruning Test")
    plt.plot([x[0] for x in pre_pruning_train], [x[1] for x in pre_pruning_train], label="Pre-pruning Train")
    plt.plot([x[0] for x in post_pruning_test], [x[1] for x in post_pruning_test], label="Post-pruning Test")
    plt.plot([x[0] for x in post_pruning_train], [x[1] for x in post_pruning_train], label="Post-pruning Train")

    plt.legend()
    plt.show()
