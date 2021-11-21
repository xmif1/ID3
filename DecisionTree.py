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

# class DecisionTree:
#     def __init__(self, dataset, attributes, target):
#         self.root = DecisionTreeNode(dataset)
