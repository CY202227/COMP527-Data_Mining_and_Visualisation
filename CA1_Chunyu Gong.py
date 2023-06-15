import numpy as np


def PerceptronTrain(data, MaxIter):
    # Initialize weights and bias
    weight = [0.0, 0.0, 0.0, 0.0]
    bias = 0
    for i in range(MaxIter):
        # Make sure the result is the same
        np.random.seed(114514)
        # Shuffle the data
        np.random.shuffle(data)
        for row in data:
            x = row[:-1]
            y = row[-1]
            a = np.dot(weight, x) + bias
            if y * a <= 0:
                weight += y * x
                bias += y
    return weight, bias


def PerceptronL2Train(data, MaxIter, reg):
    # Initialize weights and bias
    weight = [0.0, 0.0, 0.0, 0.0]
    bias = 0
    for i in range(MaxIter):
        # Make sure the result is the same
        np.random.seed(114514)
        # Shuffle the data
        np.random.shuffle(data)
        for row in data:
            x = row[:-1]
            y = row[-1]
            a = np.dot(weight, x) + bias
            if y * a <= 0:
                weight += y * x
                bias += y
    weight -= reg * weight
    return weight, bias


def PerceptronTest(data, w, b):
    num_correct = 0

    for row in data:
        x = row[:-1]
        y = row[-1]

        a = np.dot(w, x) + b

        if np.sign(a) == y:
            num_correct += 1
    accuracy = num_correct / len(data)

    return accuracy


def load_train_data(ctype1, ctype2):
    # Load data from file
    train_data = []
    with open('./data/train.data', 'r') as f:
        for line in f:
            row = line.strip().split(',')
            features = [float(x) for x in row[:-1]]  # creates a new list
            if row[-1] == ctype1:
                label = 1
            elif row[-1] == ctype2:
                label = -1  # convert label
            else:
                continue
            train_data.append(features + [label])
        train_data = np.array(train_data)
    return train_data


def load_test_data(ctype1, ctype2):
    test_data = []
    with open('./data/test.data', 'r') as f:
        for line in f:
            row = line.strip().split(',')
            features = [float(x) for x in row[:-1]]
            if row[-1] == ctype1:
                label = 1
            elif row[-1] == ctype2:
                label = -1  # convert label
            else:
                continue
            test_data.append(features + [label])
        test_data1 = np.array(test_data)
    return test_data1


class multiclass():
    def load_1vsRest_trainData(self):
        # Load data from file
        train_data = []
        with open('./data/train.data', 'r') as f:
            for line in f:
                row = line.strip().split(',')
                features = [float(x) for x in row[:-1]]
                if row[-1] == self:
                    label = 1
                elif row[-1] != self:
                    label = -1  # convert label
                else:
                    continue
                train_data.append(features + [label])
            train_data = np.array(train_data)
        return train_data

    def load_1vsRest_testData(self):
        test_data = []
        with open('./data/test.data', 'r') as f:
            for line in f:
                row = line.strip().split(',')
                features = [float(x) for x in row[:-1]]
                if row[-1] == self:
                    label = 1
                elif row[-1] != self:
                    label = -1  # convert label
                else:
                    continue
                test_data.append(features + [label])
            test_data = np.array(test_data)
        return test_data


max_iter = 20
reg = [0.01, 0.1, 1.0, 10.0, 100.0]

# Question 3
print("{:=^50s}".format("Question 3"))
print("{:=^50s}".format("Classifier for class 1 and class 2"))
# Train classifiers for class 1 and class 2
train_data_1_2 = load_train_data('class-1', 'class-2')
test_data_1_2 = load_test_data('class-1', 'class-2')
w_1_2, b_1_2 = PerceptronTrain(train_data_1_2, max_iter)
train_acc_1_2 = PerceptronTest(test_data_1_2, w_1_2, b_1_2)
test_acc_1_2 = PerceptronTest(test_data_1_2, w_1_2, b_1_2)
print(f"Train accuracy: {train_acc_1_2:.2f}")
print(f"Test accuracy: {test_acc_1_2:.2f}")

# Train classifiers for class 2 and class 3
print("{:=^50s}".format("Classifier for class 2 and class 3"))
train_data_2_3 = load_train_data('class-2', 'class-3')
test_data_2_3 = load_test_data('class-2', 'class-3')
w_2_3, b_2_3 = PerceptronTrain(train_data_2_3, max_iter)
train_acc_2_3 = PerceptronTest(train_data_2_3, w_2_3, b_2_3)
test_acc_2_3 = PerceptronTest(test_data_2_3, w_2_3, b_2_3)
print("Classifier for class 2 and class 3:")
print(f"Train accuracy: {train_acc_2_3:.2f}")
print(f"Test accuracy: {test_acc_2_3:.2f}")

# Train classifier for class 1 vs class 3
print("{:=^50s}".format("Classifier for class 1 and class 3"))
train_data_1_3 = load_train_data('class-1', 'class-3')
test_data_1_3 = load_test_data('class-1', 'class-3')
w_1_3, b_1_3 = PerceptronTrain(train_data_1_3, max_iter)
train_acc_1_3 = PerceptronTest(train_data_1_3, w_1_3, b_1_3)
test_acc_1_3 = PerceptronTest(test_data_1_3, w_1_3, b_1_3)
print("Classifier for class 1 and class 3:")
print(f"Train accuracy: {train_acc_1_3:.2f}")
print(f"Test accuracy: {test_acc_1_3:.2f}")

# Question 4
# Train classifiers for class 1 vs rest
print("{:=^50s}".format("Question 4"))

train_data_1_R = multiclass.load_1vsRest_trainData('class-1')
test_data_1_R = multiclass.load_1vsRest_testData('class-1')
w_1_R, b_1_R = PerceptronTrain(train_data_1_R, max_iter)
train_acc_1_R = PerceptronTest(train_data_1_R, w_1_R, b_1_R)
test_acc_1_R = PerceptronTest(test_data_1_R, w_1_R, b_1_R)

print("{:=^50s}".format("Classifier for class-1 vs the rest"))

print(f"Train accuracy: {train_acc_1_R:.2f}")
print(f"Test accuracy: {test_acc_1_R:.2f}")

# Train classifiers for class 2 vs rest

train_data_2_R = multiclass.load_1vsRest_trainData('class-2')
test_data_2_R = multiclass.load_1vsRest_testData('class-2')
w_2_R, b_2_R = PerceptronTrain(train_data_2_R, max_iter)
train_acc_2_R = PerceptronTest(train_data_2_R, w_2_R, b_2_R)
test_acc_2_R = PerceptronTest(test_data_2_R, w_2_R, b_2_R)

print("{:=^50s}".format("Classifier for class-2 vs the rest"))

print(f"Train accuracy: {train_acc_2_R:.2f}")
print(f"Test accuracy: {test_acc_2_R:.2f}")

# Train classifiers for class 3 vs rest

train_data_3_R = multiclass.load_1vsRest_trainData('class-3')
test_data_3_R = multiclass.load_1vsRest_testData('class-3')
w_3_R, b_3_R = PerceptronTrain(train_data_3_R, max_iter)
train_acc_3_R = PerceptronTest(train_data_3_R, w_3_R, b_3_R)
test_acc_3_R = PerceptronTest(test_data_3_R, w_3_R, b_3_R)

print("{:=^50s}".format("Classifier for class-3 vs the rest"))

print(f"Train accuracy: {train_acc_3_R:.2f}")
print(f"Test accuracy: {test_acc_3_R:.2f}")

# Question 5
# Train classifiers for class 1 vs rest
print("{:=^50s}".format("Question 5"))

train_data_1_R = multiclass.load_1vsRest_trainData('class-1')
test_data_1_R = multiclass.load_1vsRest_testData('class-1')
for i in range(5):
    w_1_2, b_1_2 = PerceptronL2Train(train_data_1_2, max_iter, reg[i])
    train_acc_1_2 = PerceptronTest(test_data_1_2, w_1_2, b_1_2)
    test_acc_1_2 = PerceptronTest(test_data_1_2, w_1_2, b_1_2)
    print("{:=^50s}".format("Classifier for class-1 vs the rest"))
    print("Set the regularisation coefficient " + str(reg[i]) + "\n")
    print(f"Train accuracy(regularisation coefficient): {train_acc_1_2:.2f}")
    print(f"Test accuracy: {test_acc_1_2:.2f}")

# Train classifiers for class 2 vs rest
train_data_2_R = multiclass.load_1vsRest_trainData('class-2')
test_data_2_R = multiclass.load_1vsRest_testData('class-2')
for i in range(5):
    w_2_3, b_2_3 = PerceptronL2Train(train_data_2_3, max_iter, reg[i])
    train_acc_2_3 = PerceptronTest(train_data_2_3, w_2_3, b_2_3)
    test_acc_2_3 = PerceptronTest(test_data_2_3, w_2_3, b_2_3)
    print("{:=^50s}".format("Classifier for class-2 vs the rest"))
    print("Set the regularisation coefficient " + str(reg[i]) + "\n")
    print(f"Train accuracy(regularisation coefficient): {train_acc_2_3:.2f}")
    print(f"Test accuracy: {test_acc_2_3:.2f}")

# Train classifiers for class 3 vs rest
train_data_3_R = multiclass.load_1vsRest_trainData('class-3')
test_data_3_R = multiclass.load_1vsRest_testData('class-3')
for i in range(5):
    w_1_3, b_1_3 = PerceptronL2Train(train_data_1_3, max_iter, reg[i])
    train_acc_1_3 = PerceptronTest(train_data_1_3, w_1_3, b_1_3)
    test_acc_1_3 = PerceptronTest(test_data_1_3, w_1_3, b_1_3)
    print("{:=^50s}".format("Classifier for class-3 vs the rest"))
    print("Set the regularisation coefficient " + str(reg[i]) + "\n")
    print(f"Train accuracy(regularisation coefficient): {train_acc_1_3:.2f}")
    print(f"Test accuracy: {test_acc_1_3:.2f}")
