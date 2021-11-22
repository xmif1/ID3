import Utilities
import copy


class DecisionTreeNode:
    def __init__(self):
        self.split_attr = None
        self.node_attr = None
        self.node_attr_val = None
        self.classification = None
        self.children = []
        self.parent = None
        self.depth = 0

    def set_parent(self, parent):
        self.parent = parent
        self.depth = parent.depth + 1


class DecisionTree:
    def __init__(self, dataset, attributes_dict, target, missing):
        self.dataset = dataset
        self.target = target
        self.missing = missing
        self.attributes_dict = attributes_dict
        self.root = self._id3(dataset, attributes_dict, target)

    def _id3(self, dataset, attributes_dict, target):
        root = DecisionTreeNode()

        most_common_val = None
        max_count = 0
        for value, count in dataset[target].value_counts().items():
            if max_count <= count:
                max_count = count
                most_common_val = value

            if count == dataset.shape[0]:
                root.classification = value
                return root

        if len(attributes_dict) == 0:
            root.classification = most_common_val
            return root

        root.split_attr = Utilities.best_attribute(dataset, attributes_dict, target)

        most_common_split_val = None
        max_count = 0
        has_missing = False
        for value, count in dataset[root.split_attr].value_counts().items():
            if value != self.missing and max_count <= count:
                max_count = count
                most_common_split_val = value
            elif value == self.missing:
                has_missing = True

        if has_missing:
            self.dataset.loc[self.dataset[root.split_attr] == self.missing, root.split_attr] = most_common_split_val

        for value in attributes_dict[root.split_attr]:
            subset = dataset.loc[dataset[root.split_attr] == value]

            if subset.shape[0] == 0:
                child = DecisionTreeNode()
                child.set_parent(root)
                child.node_attr = root.split_attr
                child.node_attr_val = value
                child.classification = most_common_val
            else:
                modified_attr_dict = copy.deepcopy(attributes_dict)
                modified_attr_dict.pop(root.split_attr)

                child = self._id3(subset, modified_attr_dict, target)
                child.set_parent(root)
                child.node_attr = root.split_attr
                child.node_attr_val = value

            root.children.append(child)

        return root

    def prune(self):
        return None

    def predict(self, predict_attr_dict):
        for pred_attr, value in predict_attr_dict.items():
            if pred_attr in self.attributes_dict:
                if value not in self.attributes_dict[pred_attr]:
                    raise ValueError("Value " + value + " is not a valid value for attribute " + pred_attr + ".")
            else:
                raise ValueError("Attribute " + pred_attr + " is not in the dataset header.")

        curr_node = self.root
        while curr_node.classification is None:
            for child in curr_node.children:
                if child.node_attr_val == predict_attr_dict[child.node_attr]:
                    curr_node = child
                    break

        return curr_node.classification

    def benchmark(self, dataset):
        n_test_samples = dataset.shape[0]
        n_positives = 0

        for _, predict_attr_dict in dataset.iterrows():
            target_value = predict_attr_dict[self.target]
            predict_attr_dict.pop(self.target)

            if self.predict(predict_attr_dict) == target_value:
                n_positives = n_positives + 1

        return n_positives / n_test_samples
