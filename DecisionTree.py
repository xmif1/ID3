import Utilities
import copy

"""
Data structure representing a node in a decision tree; maintains a number of attributes as well as provides a number of
convenience function.
"""
class DecisionTreeNode:
    def __init__(self):
        self.split_attr = None              # Attribute by which data subset is split at this node
        self.node_attr = None               # Attribute associated with this node
        self.node_attr_val = None           # Categorical value of node_attr associated with this node
        self.classification = None          # Categorical value of the target attribute associated with this node
        self.subset_target_class = None     # Most common categorical value of the target attribute associated with the
                                                # data subset at this node
        self.children = []                  # Pointers to branches associated with the categorical values of split_attr
        self.parent = None                  # Reference to the parent node (except for the root)
        self.depth = 0                      # Maintains the depth of the node from a root

    # Sets the parent reference while updating the height
    def set_parent(self, parent):
        self.parent = parent
        self.depth = parent.depth + 1

    # Removes a child node based on matches on the node_attr and node_attr_val values (which in a correct decision tree
    # constructed using ID3 should be unique).
    def remove_child(self, node):
        self.children = [c for c in self.children if (c.node_attr != node.node_attr or
                                                      c.node_attr_val != node.node_attr_val)]

    # Returns a Python generator, using which we can visit all the nodes of the subtree rooted at the node; to an extent
    # one can consider this as a 'bottom-up recursive iterator' across the tree structure.
    def traverse_subtree(self):
        for c in self.children:  # visit the nodes in each subtree rooted at a child node
            yield from c.traverse_subtree()

        yield self  # visit self


def _prune_subtree(node):
    leaf = DecisionTreeNode()
    leaf.set_parent(node.parent)
    leaf.node_attr = node.node_attr
    leaf.node_attr_val = node.node_attr_val
    leaf.subset_target_class = node.subset_target_class
    leaf.classification = node.subset_target_class

    node.parent.remove_child(node)
    node.parent.children.append(leaf)

    return leaf


class DecisionTree:
    def __init__(self, dataset, attributes_dict, target, missing):
        self.dataset = dataset
        self.target = target
        self.missing = missing
        self.attributes_dict = attributes_dict

        self.root = DecisionTreeNode()
        self.root = self._id3(self.root, dataset, attributes_dict, target)

    def _id3(self, root, dataset, attributes_dict, target):

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

        root.subset_target_class = most_common_val
        root.split_attr = Utilities.best_attribute(dataset, attributes_dict, target)

        most_common_split_val = None
        max_count = 0
        has_missing = False

        # find the most common occurring non--missing value for the splitting attribute
        for value, count in dataset[root.split_attr].value_counts().items():
            if value != self.missing and max_count <= count:  # if value exceeds current known maximum, select it instead
                max_count = count
                most_common_split_val = value
            elif value == self.missing:  # else if value is missing, set the has_missing flag to True
                has_missing = True

        # if has_missing is True and a non-missing common value is found, re-label missing data with most_common_split_val
        if has_missing and most_common_split_val is not None:
            self.dataset.loc[self.dataset[root.split_attr] == self.missing, root.split_attr] = most_common_split_val

        for value in attributes_dict[root.split_attr][0]:
            subset = dataset.loc[dataset[root.split_attr] == value]

            if subset.shape[0] == 0:
                child = DecisionTreeNode()
                child.set_parent(root)
                child.node_attr = root.split_attr
                child.node_attr_val = value
                child.classification = most_common_val
                child.subset_target_class = most_common_val
            else:
                modified_attr_dict = copy.deepcopy(attributes_dict)
                modified_attr_dict.pop(root.split_attr)

                child = DecisionTreeNode()
                child.set_parent(root)
                child.node_attr = root.split_attr
                child.node_attr_val = value

                child = self._id3(child, subset, modified_attr_dict, target)

            root.children.append(child)

        return root

    def predict(self, predict_attr_dict):
        for pred_attr, value in predict_attr_dict.items():
            if pred_attr in self.attributes_dict:
                if value not in self.attributes_dict[pred_attr][0]:
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

        if n_test_samples == 0:
            return 0
        else:
            for _, predict_attr_dict in dataset.iterrows():
                target_value = predict_attr_dict[self.target]
                predict_attr_dict.pop(self.target)

                if self.predict(predict_attr_dict) == target_value:
                    n_positives = n_positives + 1

            return n_positives / n_test_samples

    def prune_tree(self, pruning_test_set):
        prev_benchmark = self.benchmark(pruning_test_set)  # initial benchmark before pruning

        # traverse_subtree() returns an iterator which visits all the nodes in a bottom up manner
        for node in self.root.traverse_subtree():
            # if node is not the root and not a leaf node, test for pruning
            if node.parent is not None and len(node.children) != 0:
                # _prune_subtree(node) replaces the node with a leaf (which is returned) whose class is the is the most
                # common class amongst the leafs of the sub--tree rooted at node
                leaf = _prune_subtree(node)

                curr_benchmark = self.benchmark(pruning_test_set)  # benchmark the pruned decision tree
                if prev_benchmark < curr_benchmark:  # if pruning improves the benchmark, keep the pruned tree
                    prev_benchmark = curr_benchmark
                else:  # otherwise restore the node
                    node.parent.remove_child(leaf)
                    node.parent.children.append(node)

        return prev_benchmark  # return the new benchmark after pruning
