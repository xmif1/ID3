import pandas as pd
import numpy as np


def load_dataset(data_file, header_file, continuous_file, target, missing, train_frac):
    """
    :param data_file: 
    :param header_file:
    :param continuous_file:
    :param target:
    :param missing:
    :param train_frac:
    :return:
    """
    dataset = pd.read_csv(data_file, header=None)

    with open(header_file, "r") as header_file_stream:
        dataset.columns = header_file_stream.readline().rstrip().split(",")

    continuous_attr = []
    continuous_attr_thresholds = {}
    if continuous_file is not None:
        with open(continuous_file, "r") as continuous_file_stream:
            continuous_attr = continuous_file_stream.readline().rstrip().split(",")

        for attr in continuous_attr:
            if attr == target:
                raise ValueError("Target attribute cannot be continuous.")

            dataset.loc[dataset[attr] == missing, attr] = np.NaN
            dataset[attr] = dataset[attr].astype(float)

            attr_dataset = dataset[[attr, target]]
            if missing is not None:
                attr_dataset = attr_dataset.dropna(subset=[attr])

            attr_dataset = attr_dataset.sort_values(by=[attr])
            attr_dataset = attr_dataset.to_numpy()

            inflection_points = np.where(attr_dataset[:, 1][:-1] != attr_dataset[:, 1][1:])[0]
            cutoff_values = [(attr_dataset[:, 0][i] + attr_dataset[:, 0][i+1]) / 2 for i in inflection_points]

            dataset_entropy = entropy(dataset, target)
            max_information_gain = 0
            thresholded_attr_vals = []
            threshold = 0

            for c in cutoff_values:
                thresholded_c_attr_vals = []
                for value in dataset[attr]:
                    if np.isnan(value):
                        thresholded_c_attr_vals.append(missing)
                    elif value < c:
                        thresholded_c_attr_vals.append("true")
                    else:
                        thresholded_c_attr_vals.append("false")

                subset = dataset.copy()
                subset[attr] = thresholded_c_attr_vals
                c_information_gain = information_gain(subset, dataset_entropy, target, attr)
                
                if max_information_gain <= c_information_gain:
                    threshold = c
                    max_information_gain = c_information_gain
                    thresholded_attr_vals = thresholded_c_attr_vals

            dataset[attr] = thresholded_attr_vals
            continuous_attr_thresholds[attr] = threshold

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
                test = non_missing.sample(frac=(n_test_samples / non_missing.shape[0]))
                train = dataset.drop(test.index)
                train = train.loc[train[target] != missing]
        else:
            train = dataset.sample(frac=train_frac)
            test = dataset.drop(train.index)
    else:
        raise ValueError("Fraction split is not in the range 0 < f <= 1 as required.")

    attributes_dict = {}
    target_found = False
    for attribute in dataset.columns:
        if attribute != target:
            values = []
            for v in dataset[attribute].unique():
                if v != missing:
                    values.append(v)

            if attribute in continuous_attr:
                attributes_dict[attribute] = [values, continuous_attr_thresholds[attribute]]
            else:
                attributes_dict[attribute] = [values, None]
        else:
            target_found = True

    if not target_found:
        raise ValueError("Specified target attribute is not in the header file.")

    return train, test, attributes_dict


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
