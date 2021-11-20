class DecisionTreeNode:
    def __init__(self, dataset, parent=None):
        self.dataset = dataset
        self.split_attr = None
        self.node_attr = None
        self.node_attr_val = None
        self.classification = None
        self.children = None
        self.parent = parent
        self.depth = 0

        if parent is not None:
            self.depth = parent.depth + 1


# class DecisionTree:
#     def __init__(self, dataset, attributes, target):
#         self.root = DecisionTreeNode(dataset)
