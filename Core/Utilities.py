import pandas as pd
import numpy as np

"""
Convenience function for loading a data set from a CSV file into a Panda DataFrame, and preparing it for use with a 
decision tree while also extracting meta-data about the data set. In particular, the function is responsible for:
    1.  Splitting the data in training and test subsets, such that the test set has no missing data (i.e. we can carry
        out predictions on it)
    2.  Determining a threshold value with which continuous valued attributes are discretised, and reflecting that
        discretisation in the imported dataset
    3.  Generates a dictionary with an entry associated with each attribute; the values stored are a list L of size 2:
            a. L[0] returns a list of all the categorical values the attribute takes
            b. L[1] returns None if the attribute is categorical, or else returns the threshold value with which the
               continuous attribute was discretised.
"""
def load_dataset(data_file, header_file, continuous_file, target, missing, train_frac):
    dataset = pd.read_csv(data_file, header=None)  # Load data from CSV file

    with open(header_file, "r") as header_file_stream:  # Add column names to DataFrame from attributes header file
        dataset.columns = header_file_stream.readline().rstrip().split(",")

    continuous_attr = []  # List storing all the continuous valued attributes
    continuous_attr_thresholds = {}  # List storing the threshold value with which each continuous attribute is discretised
    if continuous_file is not None: # If continuous attributes specified...
        # Populate continuous_attr with the specified attributes
        with open(continuous_file, "r") as continuous_file_stream:
            continuous_attr = continuous_file_stream.readline().rstrip().split(",")

        # For each of the specified continuous attributes, determine the best threshold value (the one which maximises
        # information gain) and discretise using that value.
        for attr in continuous_attr:
            if attr == target:  # In case that target is specified as continuous, return error.
                raise ValueError("Target attribute cannot be continuous.")

            # Since continuous data is real valued and missing data can be represented using any alphanumeric string,
            # before we represent the attribute values as a float type we must convert the missing value to np.NaN, as
            # otherwise the conversion cannot be done if say that missing data is represented by "?".
            dataset.loc[dataset[attr] == missing, attr] = np.NaN
            dataset[attr] = dataset[attr].astype(float)

            # Construct a subset consisting solely of the chosen continuous attribute and the target attribute, without
            # any missing data.
            attr_dataset = dataset[[attr, target]]
            if missing is not None:
                attr_dataset = attr_dataset.dropna(subset=[attr])

            # Then sort by the continuous attribute and convert to a Numpy array.
            attr_dataset = attr_dataset.sort_values(by=[attr])
            attr_dataset = attr_dataset.to_numpy()

            # It is known that, for a sorted dataset as constructed above, the information gain is maximised at some pt
            # between changes in the target attribute values. Hence we find all such points of inflection, and calculate
            # the average value of the continuous valued attribute at which these inflections in the target occur.
            inflection_points = np.where(attr_dataset[:, 1][:-1] != attr_dataset[:, 1][1:])[0]
            threshold_values = [(attr_dataset[:, 0][i] + attr_dataset[:, 0][i+1]) / 2 for i in inflection_points]

            dataset_entropy = entropy(dataset, target)  # Calculate the entropy of the entire dataset
            max_information_gain = 0                    # Will store the maximum information gain at the cutoff points
            thresholded_attr_vals = []                  # Will store the discretised continuous attribute column
            threshold = 0                               # Will store the chosen threshold by which the continuous
                                                            # attribute is discretised

            # For every possible threshold value...
            for t in threshold_values:
                # Discretise the data
                thresholded_c_attr_vals = []
                for value in dataset[attr]:
                    if np.isnan(value):  # if NaN, replace with the missing simple
                        thresholded_c_attr_vals.append(missing)
                    elif value < t:  # if less than the threshold c, replace with "true"
                        thresholded_c_attr_vals.append("true")
                    else:  # if greater than the threshold c, replace with "false"
                        thresholded_c_attr_vals.append("false")

                # Copy the original dataset, substitute the disretised continuous attribute and calculate the associated
                # information gain.
                subset = dataset.copy()
                subset[attr] = thresholded_c_attr_vals
                c_information_gain = information_gain(subset, dataset_entropy, target, attr)

                # If the information gain associated with the threshold is greater than the information gain associated
                # with any of the previously tested threshold values, then update accordingly.
                if max_information_gain <= c_information_gain:
                    threshold = t
                    max_information_gain = c_information_gain
                    thresholded_attr_vals = thresholded_c_attr_vals

            # Discretise the continuous dataset using the threshold greedily chosen to maximise the information gain
            dataset[attr] = thresholded_attr_vals
            continuous_attr_thresholds[attr] = threshold

    if 0 < train_frac <= 1:
        if missing is not None:  # if missing label is specified
            n_test_samples = dataset.shape[0] * (1 - train_frac)    # number of test samples desired

            # filter out missing data based on passed label, column by column starting with the target...
            non_missing = dataset.loc[dataset[target] != missing]
            for attr in non_missing.columns:    # ...and then filtering out attribute by attribute
                if attr != target:
                    non_missing = non_missing.loc[non_missing[attr] != missing]

            if non_missing.shape[0] < n_test_samples:  # if not enough filtered data to generate testing set of the
                                                       # desired size, throw an exception
                raise ValueError("Not enough samples without missing data to generate training set of specified size.")
            else:
                # otherwise split the data set, such that the test set has no missing data, but the training set does
                # since this can be dealt with later on using the ID3 algorithm
                test = non_missing.sample(frac=(n_test_samples / non_missing.shape[0]))
                train = dataset.drop(test.index)
                train = train.loc[train[target] != missing]  # note however that target cannot have missing data
        else:  # else if no missing label specified, simply split using the given fraction
            train = dataset.sample(frac=train_frac)
            test = dataset.drop(train.index)
    else:
        raise ValueError("Fraction split is not in the range 0 < f <= 1 as required.")

    # Construct attributes dict as described above
    attributes_dict = {}
    target_found = False  # Verify that target column actually is in the dataset
    for attribute in dataset.columns:
        if attribute != target:  # For each attribute which is not the target...
            # Find all the distinct values (incl. missing) that the attribute can take
            values = []
            for v in dataset[attribute].unique():
                if v != missing:
                    values.append(v)

            # If attribute is continuous, maintain the thresholding value...
            if attribute in continuous_attr:
                attributes_dict[attribute] = [values, continuous_attr_thresholds[attribute]]
            else:
                attributes_dict[attribute] = [values, None]
        else:
            target_found = True

    if not target_found:  # If target attribute not in dataset, throw error...
        raise ValueError("Specified target attribute is not in the header file.")

    return train, test, attributes_dict  # return prepared training and test sets, along with the attributes dictionary


# Utility function for computing the entropy of a dataset based on the target column specified.
def entropy(dataset, target):
    # Count the number of occurrences of each value the target attribute takes in the dataset
    counts = [c for _, c in dataset[target].value_counts().items()]

    # Normalise the counts by the size of the size of the dataset, yielding a distribution of how probable each target
    # value is likely to occur
    probabilities = np.divide(counts, dataset.shape[0])

    # Compute the entropy, which is - the log2 sum of the probabilities weighted by the probabilities themselves
    return -1 * np.sum([p * log2_p for p, log2_p in zip(probabilities, np.log2(probabilities))])


# Utility function that computes the information gain for a given attribute
def information_gain(dataset, dataset_entropy, target, attribute):
    # InformationGain(Dataset, Attr) = Entropy(Dataset) - ConditionalEntropy(Dataset | Attr)

    conditional_entropy = 0  # Initial value for the conditional entropy...

    for value in dataset[attribute].unique():  # For each value attribute takes in the dataset...
        # Filter the dataset to contain only those entries where attribute == value
        subset = dataset.loc[dataset[attribute] == value]

        # Aggregate the entropy of the filtered dataset weighted by the size of the filtered dataset
        conditional_entropy = conditional_entropy + (subset.shape[0] * entropy(subset, target))

    # Normalise by the size of the original dataset
    conditional_entropy = conditional_entropy / dataset.shape[0]

    return dataset_entropy - conditional_entropy  # Compute and return the information gain


# Utility function that greedily find the attribute which maximises the information gain.
def best_attribute(dataset, attributes_dict, target):
    dataset_entropy = entropy(dataset, target)  # Find the entropy of the entire dataset.
    max_information_gain = 0  # Current maximum information gain with respect to all the attributes in attributes_dict
    max_ig_attr = None  # Attribute corresponding to the current maximum information gain

    for attr in attributes_dict:  # For each attribute in attributes_dict...
        if attr != target:  # ...which is not the target...
            # Compute the information gain
            ig = information_gain(dataset, dataset_entropy, target, attr)

            # If the information gain computed is greater than the globally found information gain, adjust...
            if max_information_gain <= ig:
                max_information_gain = ig
                max_ig_attr = attr

    return max_ig_attr  # Return the attribute that maximises the information gain
