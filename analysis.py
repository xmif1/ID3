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
    pre_pruning_height = []
    post_pruning_height = []

    for f in range(1, 8):
        frac = f / 10.0
        try:
            train, test, attributes_dict = Utilities.load_dataset(args["data"], args["header"], args["continuous"],
                                                                  args["target"], args["missing"], frac)

            pruning_test_set = test.sample(frac=(1/3))
            benchmark_test_set = test.drop(pruning_test_set.index)

            decisionTree = DecisionTree(train, attributes_dict, args["target"], args["missing"])
            train_non_missing = decisionTree.dataset.loc[decisionTree.dataset[args["target"]] != args["missing"]]

            for attr in train_non_missing.columns:
                if attr != args["target"]:
                    train_non_missing = train_non_missing.loc[train_non_missing[attr] != args["missing"]]

            pre_pruning_test.append([frac, decisionTree.benchmark(benchmark_test_set)])
            pre_pruning_train.append([frac, decisionTree.benchmark(train_non_missing)])
            pre_pruning_height.append([frac, max([node.depth for node in decisionTree.root.traverse_subtree()])])

            decisionTree.prune_tree(pruning_test_set)

            post_pruning_test.append([frac, decisionTree.benchmark(benchmark_test_set)])
            post_pruning_train.append([frac, decisionTree.benchmark(train_non_missing)])
            post_pruning_height.append([frac, max([node.depth for node in decisionTree.root.traverse_subtree()])])
        except ValueError as ve:
            print(ve)
            print("Exiting...")
            exit(1)

    fig1, (ax1) = plt.subplots(1, 1)
    fig1.suptitle("Overfitting Analysis Pre and Post Pruning")

    ax1.plot([x[0] for x in pre_pruning_test], [x[1] for x in pre_pruning_test], label="Pre-pruning Test")
    ax1.plot([x[0] for x in pre_pruning_train], [x[1] for x in pre_pruning_train], label="Pre-pruning Train")
    ax1.plot([x[0] for x in post_pruning_test], [x[1] for x in post_pruning_test], label="Post-pruning Test")
    ax1.plot([x[0] for x in post_pruning_train], [x[1] for x in post_pruning_train], label="Post-pruning Train")

    ax1.legend()

    fig2, (ax2) = plt.subplots(1, 1)
    fig2.suptitle("Decision Tree Height Analysis Pre and Post Pruning")

    ax2.plot([x[0] for x in pre_pruning_height], [x[1] for x in pre_pruning_height], label="Pre-pruning height")
    ax2.plot([x[0] for x in post_pruning_height], [x[1] for x in post_pruning_height], label="Post-pruning height")

    ax2.legend()

    plt.show()
