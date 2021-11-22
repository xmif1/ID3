from DecisionTree import DecisionTree
import Utilities

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-D", "--data", required=True, help="File path to a .data file")
ap.add_argument("-H", "--header", required=True, help="File path to a .header file, corresponding to the .data file")
ap.add_argument("-t", "--target", required=True, help="Target attribute name appearing in the .header file")
ap.add_argument("-f", "--fraction-split", required=True, type=float,
                help="Fraction split into training (f) and test data (1-f)")
ap.add_argument("-m", "--missing", required=False, help="Missing attribute value flag")
args = vars(ap.parse_args())


def test_decision_tree(decision_tree, test_dataset):
    n_test_samples = test_dataset.shape[0]
    n_positives = 0

    for _, predict_attr_dict in test_dataset.iterrows():
        target_value = predict_attr_dict[decision_tree.target]
        predict_attr_dict.pop(decision_tree.target)

        if decision_tree.predict(predict_attr_dict) == target_value:
            n_positives = n_positives + 1

    print("Testing set size = " + str(n_test_samples) + ", % correct predictions = " +
          str((n_positives / n_test_samples) * 100))


if __name__ == "__main__":
    try:
        train, test, attribute_dict = Utilities.load_dataset(args["data"], args["header"], args["target"],
                                                             args["missing"], train_frac=args["fraction_split"])

        decisionTree = DecisionTree(train, attribute_dict, args["target"], args["missing"])
        test_decision_tree(decisionTree, test)
    except ValueError as ve:
        print(ve)
        print("Exiting...")
        exit(1)
