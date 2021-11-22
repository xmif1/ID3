from DecisionTree import DecisionTree
import Utilities

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-D", "--data", required=True, help="File path to a .data file")
ap.add_argument("-H", "--header", required=True, help="File path to a .header file, corresponding to the .data file")
ap.add_argument("-t", "--target", required=True, help="Target attribute name appearing in the .header file")
ap.add_argument("-f", "--fraction-split", required=True, type=float,
                help="Fraction split into training (f) and test data (1-f)")
args = vars(ap.parse_args())


if __name__ == "__main__":
    try:
        train, test, attribute_dict = Utilities.load_dataset(args["data"], args["header"], args["target"],
                                                             train_frac=args["fraction_split"])

        decisionTree = DecisionTree(train, attribute_dict, args["target"])
        # pred_attr_dict = {"outlook": "overcast",
        #                   "temp": "hot",
        #                   "humidity": "high",
        #                   "wind": "weak"}
        # print(decisionTree.predict(pred_attr_dict))
    except ValueError as ve:
        print(ve)
        print("Exiting...")
        exit(1)
