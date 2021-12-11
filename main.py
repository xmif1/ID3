from DecisionTree import DecisionTree
import Utilities

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

        # Construct a decision tree using the ID3 algorithm
        decisionTree = DecisionTree(train, attributes_dict, args["target"], args["missing"])

        # Filter out any training data with missing values, for the purposes of prediction benchmarking
        train_non_missing = decisionTree.dataset.loc[decisionTree.dataset[args["target"]] != args["missing"]]

        for attr in train_non_missing.columns:
            if attr != args["target"]:
                train_non_missing = train_non_missing.loc[train_non_missing[attr] != args["missing"]]

        # Benchmark the decision tree on the training and test sets, before any pruning for over--fitting minimisation
        print("Pre-pruning Testing benchmark: " + str(decisionTree.benchmark(test)))
        print("Pre-pruning Training benchmark: " + str(decisionTree.benchmark(train_non_missing)))

        # Prune the decision tree to minimise over--fitting
        decisionTree.prune_tree(test)

        # Benchmark the decision tree on the training and test sets, after pruning for over--fitting minimisation
        print("Post-pruning Testing benchmark: " + str(decisionTree.benchmark(test)))
        print("Post-pruning Training benchmark: " + str(decisionTree.benchmark(train_non_missing)))
    except ValueError as ve:
        print(ve)
        print("Exiting...")
        exit(1)
