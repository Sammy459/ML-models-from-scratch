import random
import numpy as np


def gini_index(x):
    count = [0, 0, 0]
    for i in x:
        count[i] += 1
    total = len(x)
    probab = [c / total for c in count]
    gini = 0
    for i in probab:
        gini += i * (1 - i)
    return gini


def best_split(X, Y):
    best_index = 784
    best_split_dim = -1
    best_split_value = -1
    dimensions = X.shape[0]
    for i in range(dimensions):
        value = np.mean(X[i, :])
        left_indices = []
        right_indices = []
        for j in range(X.shape[1]):
            if X[i, j] <= value:
                left_indices.append(j)
            else:
                right_indices.append(j)
        left_gini = gini_index(Y[left_indices])
        right_gini = gini_index(Y[right_indices])
        gini = (len(left_indices) / len(Y)) * left_gini + (len(right_indices) / len(Y)) * right_gini
        if gini < best_index:
            best_index = gini
            best_split_dim = i
            best_split_value = value
    return best_split_dim, best_split_value, best_index


def spilt_data(X, Y, dim, value):
    left1 = []
    right1 = []
    for i in range(X.shape[1]):
        if X[dim, i] <= value:
            left1.append(i)
        else:
            right1.append(i)
    left = X[:, left1]
    right = X[:, right1]
    left_labels = Y[left1]
    right_labels = Y[right1]
    return left, right, left_labels, right_labels


def decision_tree_left(X, Y, current_depth):
    if current_depth == 2 or len(np.unique(Y)) == 1:
        counts = [0, 0, 0]
        for i in Y:
            counts[i] += 1
        indice = np.argmax(counts)
        return {'class': indice}

    best_split_dim, best_split_value, best_index = best_split(X, Y)
    left, right, left_labels, right_labels = spilt_data(X, Y, best_split_dim, best_split_value)
    left_tree = decision_tree_left(left, left_labels, current_depth + 1)
    counts = [0, 0, 0]
    for i in right_labels:
        counts[i] += 1
    indice = np.argmax(counts)
    right_tree = {'class': indice}
    return {'splitDim': best_split_dim,
            'splitValue': best_split_value,
            'left': left_tree,
            'right': right_tree}


def decision_tree_right(X, Y, current_depth):
    if current_depth == 2 or len(np.unique(Y)) == 1:
        counts = [0, 0, 0]
        for i in Y:
            counts[i] += 1
        indice = np.argmax(counts)
        return {'class': indice}

    best_split_dim, best_split_value, best_index = best_split(X, Y)
    left, right, left_labels, right_labels = spilt_data(X, Y, best_split_dim, best_split_value)
    right_tree = decision_tree_right(right, right_labels, current_depth + 1)
    counts = [0, 0, 0]
    for i in left_labels:
        counts[i] += 1
    indice = np.argmax(counts)
    left_tree = {'class': indice}
    return {'splitDim': best_split_dim,
            'splitValue': best_split_value,
            'left': left_tree,
            'right': right_tree}


dataset = np.load('mnist.npz')
x = dataset['x_train']
y = dataset['y_train']
x1 = dataset['x_test']
y1 = dataset['y_test']
# x=x/255
classes = [0, 1, 2]
x2 = x.reshape(x.shape[0], -1)
array1 = []
array3 = []
for i in classes:
    classIndices = np.where(y == i)[0]
    for j in classIndices:
        array1.append(x2[j])
        array3.append(y[j])
array2 = np.array(array1)
array4 = np.array(array3)
X = array2.T
Y = array4.T
X_centered = X - np.mean(X, axis=1, keepdims=True)
covariance = np.dot(X_centered, X_centered.T) / (X.shape[1] - 1)
eigenvalues, eigenvectors = np.linalg.eigh(covariance)
sorting = np.argsort(eigenvalues)[::-1]
U = eigenvectors[:, sorting]
U_p = U[:, :10]
Y_1 = U_p.T @ X_centered
current_depth = 0
a = random.randint(0, 1)
if a == 0:
    tree = decision_tree_right(Y_1, Y, current_depth)
else:
    tree = decision_tree_left(Y_1, Y, current_depth)
x3 = x1.reshape(x1.shape[0], -1)
array5 = []
array6 = []
for i in classes:
    classIndices = np.where(y1 == i)[0]
    for j in classIndices:
        array5.append(x3[j])
        array6.append(y1[j])
array7 = np.array(array5)
array8 = np.array(array6)
X1 = array7.T
Y1 = array8.T
Y_2 = U_p.T @ (X1 - np.mean(X, axis=1, keepdims=True))

predictions = []
for i in range(Y_2.shape[1]):
    current = tree
    while 'class' not in current:
        if Y_2[current['splitDim'], i] <= current['splitValue']:
            current = current['left']
        else:
            current = current['right']
    predictions.append(current['class'])
predictions = np.array(predictions)

accuracy = 0
for i in range(len(predictions)):
    if predictions[i] == Y1[i]:
        accuracy += 1
print("Decision tree without bagging")
print("Accuracy:", accuracy / len(predictions))

print("Class-wise accuracy:")
for classLabel in classes:
    indices = []
    for i in range(len(Y1)):
        if Y1[i] == classLabel:
            indices.append(i)
    accuracy1 = 0
    for i in indices:
        if predictions[i] == Y1[i]:
            accuracy1 += 1
    accuracy1 = accuracy1 / len(indices)
    print("Class " + str(classLabel) + " : " + str(accuracy1))
print()
d1 = []
d2 = []
d3 = []
d4 = []
d5 = []
d1_2 = []
d2_2 = []
d3_2 = []
d4_2 = []
d5_2 = []
a1 = np.random.choice(len(Y_1[0]), size=len(Y_1[0]), replace=True)
a2 = np.random.choice(len(Y_1[0]), size=len(Y_1[0]), replace=True)
a3 = np.random.choice(len(Y_1[0]), size=len(Y_1[0]), replace=True)
a4 = np.random.choice(len(Y_1[0]), size=len(Y_1[0]), replace=True)
a5 = np.random.choice(len(Y_1[0]), size=len(Y_1[0]), replace=True)
for(i, j, k, l, m) in zip(a1, a2, a3, a4, a5):
    d1.append(Y_1[:, i])
    d1_2.append(Y[i])
    d2.append(Y_1[:, j])
    d2_2.append(Y[j])
    d3.append(Y_1[:, k])
    d3_2.append(Y[k])
    d4.append(Y_1[:, l])
    d4_2.append(Y[l])
    d5.append(Y_1[:, m])
    d5_2.append(Y[m])
d1 = np.array(d1).T
d2 = np.array(d2).T
d3 = np.array(d3).T
d4 = np.array(d4).T
d5 = np.array(d5).T
d1_2 = np.array(d1_2)
d2_2 = np.array(d2_2)
d3_2 = np.array(d3_2)
d4_2 = np.array(d4_2)
d5_2 = np.array(d5_2)
current_depth = 0
if a == 0:
    tree1 = decision_tree_right(d1, d1_2, current_depth)
    tree2 = decision_tree_right(d2, d2_2, current_depth)
    tree3 = decision_tree_right(d3, d3_2, current_depth)
    tree4 = decision_tree_right(d4, d4_2, current_depth)
    tree5 = decision_tree_right(d5, d5_2, current_depth)
else:
    tree1 = decision_tree_left(d1, d1_2, current_depth)
    tree2 = decision_tree_left(d2, d2_2, current_depth)
    tree3 = decision_tree_left(d3, d3_2, current_depth)
    tree4 = decision_tree_left(d4, d4_2, current_depth)
    tree5 = decision_tree_left(d5, d5_2, current_depth)

trees = [tree1, tree2, tree3, tree4, tree5]
predictions1 = []
for i in range(Y_2.shape[1]):
    temp = []
    flag1 = -1
    for j in trees:
        current = j
        while 'class' not in current:
            if Y_2[current['splitDim'], i] <= current['splitValue']:
                current = current['left']
            else:
                current = current['right']
        temp.append(current['class'])
    counts = [0, 0, 0]
    for j in temp:
        counts[j] += 1
    for j in range(3):
        if counts[j] >= 3:
            flag1 = 0
            predictions1.append(j)
            break
    if flag1 == 0:
        continue
    for j in range(3):
        if counts[j] == 2:
            predictions1.append(j)
            break


predictions1 = np.array(predictions1)
accuracy = 0
for i in range(len(predictions1)):
    if predictions1[i] == Y1[i]:
        accuracy += 1
print("Decision tree with bagging")
print("Accuracy:", accuracy / len(predictions1))

print("Class-wise accuracy:")
for classLabel in classes:
    indices=[]
    for i in range(len(Y1)):
        if Y1[i] == classLabel:
            indices.append(i)
    accuracy1 = 0
    for i in indices:
        if predictions1[i] == Y1[i]:
            accuracy1 += 1
    accuracy1=accuracy1/len(indices)
    print("Class " + str(classLabel) + " : " + str(accuracy1))
