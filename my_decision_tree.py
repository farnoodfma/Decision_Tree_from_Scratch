import numpy as np
import pandas as pd
import random


random.seed(1)

data_frame = pd.read_csv(r"The address of the dataset")


def label_column(data, data_column):
    data["label"] = data[f"{data_column}"]
    data = data.drop(f"{data_column}", axis=1)
    return data


data_frame = label_column(data_frame, "Survived")
data_frame = data_frame.drop(["PassengerId", "Name", "Ticket", "Cabin", "Embarked"], axis=1)
# data_frame["Embarked"] = data_frame["Embarked"].astype(str)
header = list(data_frame.columns)
print(header)


def dataset_split(data, test_size, random_seed=1):
    random.seed(random_seed)
    test_size = round(test_size * len(data))
    indices = data.index.tolist()

    test_indices = random.sample(population=indices, k=test_size)

    test_data_frame = data.loc[test_indices]
    train_data_frame = data.drop(test_indices)

    return train_data_frame, test_data_frame


train_data_frame, test_data_frame = dataset_split(data_frame, test_size=0.3, random_seed=2)
train_data = train_data_frame.values
test_data = test_data_frame.values
unique, count = np.unique(train_data[:, -1], return_counts=True)


def is_categorical(value, num_unique_values):
    num_unique_values_limit = 10
    return isinstance(value, str) or len(num_unique_values) <= num_unique_values_limit


class Condition:

    def __init__(self, column, value):
        self.column = column
        self.value = value
        self.num_unique = np.unique(train_data[:, self.column])

    def compare(self, example):
        val = example[self.column]
        if is_categorical(val, self.num_unique):
            return val == self.value
        else:
            return val >= self.value

    def __repr__(self):

        condition_sign = ">="
        if is_categorical(self.value, self.num_unique):
            condition_sign = "=="
        return f"Is {header[self.column]} {condition_sign} {str(self.value)}"


def split_dataset(rows, condition):
    true_rows, false_rows = [], []

    for row in rows:
        if condition.compare(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def entropy(data):
    data_list = np.array(data)
    label_col = data_list[:, -1]
    _, counts = np.unique(label_col, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
    return entropy


def information_gain(data_left, data_right, parent_entropy):
    num_data_points = len(data_left) + len(data_right)

    p_data_left = len(data_left) / num_data_points
    p_data_right = 1 - p_data_left

    overall_entropy = (p_data_left * entropy(data_left)) + (p_data_right * entropy(data_right))
    information_gain = parent_entropy - overall_entropy
    return information_gain


def find_best_split(data):
    data = np.array(data)
    best_gain = 0
    best_condition = None

    current_entropy = entropy(data)
    num_features = len(data[0]) - 1

    for col in range(num_features):
        values = np.unique(data[:, col])

        for val in values:
            condition = Condition(col, val)

            true_data, false_data = split_dataset(data, condition)

            if len(true_data) == 0 or len(false_data) == 0:
                continue

            info_gain = information_gain(true_data, false_data, current_entropy)

            if info_gain >= best_gain:
                best_gain, best_condition = info_gain, condition

    return best_gain, best_condition


class Final_Node:

    def __init__(self, data):
        data = np.array(data)
        counts = {}
        unique_val, num_unqiue = np.unique(data[:, -1], return_counts=True)
        for index in range(len(unique_val)):
            counts[unique_val[index]] = num_unqiue[index]

        self.predictions = counts


class Decision_Node:

    def __init__(self, condition, true_branch, false_branch):
        self.condition = condition
        self.true_branch = true_branch
        self.false_branch = false_branch


def decision_tree(data, max_dep, counter=0):
    gain, condition = find_best_split(data)
    counter += 1

    if gain == 0 or counter == max_dep:
        return Final_Node(data)

    true_data, false_data = split_dataset(data, condition)

    true_branch = decision_tree(true_data, max_dep, counter)
    false_branch = decision_tree(false_data, max_dep, counter)

    return Decision_Node(condition, true_branch, false_branch)


def print_tree(node, spacing=""):
    if isinstance(node, Final_Node):
        print("" + "Answer is", node.predictions)
        return

    print(" " + str(node.condition))

    print(" " + '--> True: ')
    print_tree(node.true_branch, spacing + "  ")

    print(spacing + '--> False: ')
    print_tree(node.false_branch, spacing + "  ")


my_tree = decision_tree(train_data, max_dep=4)
print_tree(my_tree)


def prediction(data, node):
    if isinstance(node, Final_Node):
        return node.predictions

    if node.condition.compare(data):
        return prediction(data, node.true_branch)
    else:
        return prediction(data, node.false_branch)


def print_final_node(counts):
    total = sum(counts.values())
    probs = {}
    for temp in counts.keys():
        probs[temp] = str(int(counts[temp] / total * 100)) + "%"
    return probs


prediction(train_data[6], my_tree)
print(print_final_node(prediction(train_data[6], my_tree)))


classified_item = []
temp_list = []

for item in test_data:
    print(f"True Value:{item[-1]} Answer: {print_final_node(prediction(item,my_tree))}")
    classified_item.append(print_final_node(prediction(item,my_tree)).values())



temp_list = [ele for ele in classified_item if len(ele) == 1]
my_tree_accuracy = (len(temp_list)/len(classified_item) ) * 100
print(str(my_tree_accuracy) + "%")

