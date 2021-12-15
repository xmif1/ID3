from Core.DecisionTree import DecisionTree
from Core import Utilities

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-D", "--data", required=True, help="File path to a .data file")
ap.add_argument("-H", "--header", required=True, help="File path to a .header file, corresponding to the .data file")
ap.add_argument("-C", "--continuous", required=False, help="File path to a .continuous file, specifying the continuous "
                                                           "data attributes in the corresponding .header file")
ap.add_argument("-t", "--target", required=True, help="Target attribute name appearing in the .header file")
ap.add_argument("-f", "--fraction-split", required=True, type=float,
                help="Fraction split into training (f) and test data (1-f)")
ap.add_argument("-m", "--missing", required=False, help="Missing attribute value flag")
args = vars(ap.parse_args())


if __name__ == "__main__":
    try:
        # Load dataset and prepare it into training and testing subsets, along with any missing data manipulation and
        # continuous attribute discretisation. Also prepare dictionary of attributes and their values.
        train, test, attributes_dict = Utilities.load_dataset(args["data"], args["header"], args["continuous"],
                                                              args["target"], args["missing"], args["fraction_split"])
        pruning_test_set = test.sample(frac=(2/3))
        benchmark_test_set = test.drop(pruning_test_set.index)

        # Construct a decision tree using the ID3 algorithm
        decisionTree = DecisionTree(train, attributes_dict, args["target"], args["missing"])

        # Filter out any training data with missing values, for the purposes of prediction benchmarking
        train_non_missing = decisionTree.dataset.loc[decisionTree.dataset[args["target"]] != args["missing"]]

        for attr in train_non_missing.columns:
            if attr != args["target"]:
                train_non_missing = train_non_missing.loc[train_non_missing[attr] != args["missing"]]

        # Benchmark the decision tree on the training and test sets, before any pruning for over--fitting minimisation
        print("Pre-pruning Testing benchmark: " + str(decisionTree.benchmark(benchmark_test_set)))
        print("Pre-pruning Training benchmark: " + str(decisionTree.benchmark(train_non_missing)))

        n_node_pre_pruning = 0
        for _ in decisionTree.root.traverse_subtree():
            n_node_pre_pruning = n_node_pre_pruning + 1
        print("Pre-pruning number of nodes: " + str(n_node_pre_pruning))

        # Prune the decision tree to minimise over--fitting
        decisionTree.prune_tree(pruning_test_set)

        # Benchmark the decision tree on the training and test sets, after pruning for over--fitting minimisation
        print("Post-pruning Testing benchmark: " + str(decisionTree.benchmark(benchmark_test_set)))
        print("Post-pruning Training benchmark: " + str(decisionTree.benchmark(train_non_missing)))

        n_node_post_pruning = 0
        for _ in decisionTree.root.traverse_subtree():
            n_node_post_pruning = n_node_post_pruning + 1
        print("Post-pruning number of nodes: " + str(n_node_post_pruning))
        print("% change in number of nodes: " + str((1 - (n_node_post_pruning / n_node_pre_pruning)) * 100) + "%")
    except ValueError as ve:
        print(ve)
        print("Exiting...")
        exit(1)
