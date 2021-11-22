import pandas as pd
import numpy as np


def load_dataset(data_file, header_file, target, missing, train_frac=0.7):
    dataset = pd.read_csv(data_file, header=None)

    with open(header_file, "r") as header_file_stream:
        dataset.columns = header_file_stream.readline().rstrip().split(",")

    if 0 < train_frac <= 1:
        if missing is not None and train_frac != 1:
            n_test_samples = dataset.shape[0] * (1 - train_frac)
            non_missing = dataset.loc[dataset[target] != missing]

            for attr in non_missing.columns:
                if attr != target:
                    non_missing = non_missing.loc[non_missing[attr] != missing]

            if non_missing.shape[0] < n_test_samples:
                raise ValueError("Not enough samples without missing data to generate training set of specified size.")
            else:
                # remove random_state once debugging is over
                test = non_missing.sample(frac=(n_test_samples / non_missing.shape[0]), random_state=0)
                train = dataset.drop(test.index)
                train = train.loc[train[target] != missing]
        else:
            train = dataset.sample(frac=train_frac, random_state=0)  # remove random_state once debugging is over
            test = dataset.drop(train.index)
    else:
        raise ValueError("Fraction split is not in the range 0 < f <= 1 as required.")

    attribute_dict = {}
    target_found = False
    for attribute in dataset.columns:
        if attribute != target:
            values = []
            for v in dataset[attribute].unique():
                if v != missing:
                    values.append(v)

            attribute_dict[attribute] = values
        else:
            target_found = True

    if not target_found:
        raise ValueError("Specified target attribute is not in the header file.")

    return train, test, attribute_dict


def entropy(dataset, target):
    counts = [c for _, c in dataset[target].value_counts().items()]
    probabilities = np.divide(counts, dataset.shape[0])

    return -1 * np.sum([p * log2_p for p, log2_p in zip(probabilities, np.log2(probabilities))])


def information_gain(dataset, dataset_entropy, target, attribute):
    conditional_entropy = 0
    for value in dataset[attribute].unique():
        subset = dataset.loc[dataset[attribute] == value]
        conditional_entropy = conditional_entropy + (subset.shape[0] * entropy(subset, target))

    conditional_entropy = conditional_entropy / dataset.shape[0]

    return dataset_entropy - conditional_entropy


def best_attribute(dataset, attributes_dict, target):
    dataset_entropy = entropy(dataset, target)
    max_information_gain = 0
    max_ig_attr = None

    for attr in attributes_dict:
        if attr != target:
            ig = information_gain(dataset, dataset_entropy, target, attr)
            if max_information_gain <= ig:
                max_information_gain = ig
                max_ig_attr = attr

    return max_ig_attr
