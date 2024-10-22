import numpy as np
import matplotlib.pyplot as plt

dataset = np.load('mnist.npz')
x = dataset['x_train']
y = dataset['y_train']
x1 = dataset['x_test']
y1 = dataset['y_test']
classes = [0, 1]
x2 = x.reshape(x.shape[0], -1)
x3 = x1.reshape(x1.shape[0], -1)
array1 = []
array2 = []
array3 = []
array4 = []
counter_0 = 0
counter_1 = 0
for i in range(len(y)):
    if y[i] == 0:
        if counter_0 < 1000:
            array3.append(x2[i])
            array4.append(-1)
            counter_0 += 1
        else:
            array1.append(x2[i])
            array2.append(-1)
    elif y[i] == 1:
        if counter_1 < 1000:
            array3.append(x2[i])
            array4.append(1)
            counter_1 += 1
        else:
            array1.append(x2[i])
            array2.append(1)
X_test_final = []
Y_test_final = []
for i in range(len(y1)):
    if y1[i] == 0:
        X_test_final.append(x3[i])
        Y_test_final.append(-1)
    elif y1[i] == 1:
        X_test_final.append(x3[i])
        Y_test_final.append(1)

X_train = np.array(array1).T
Y_train = np.array(array2)
X_test = np.array(array3).T
Y_test = np.array(array4)
X_test_final = np.array(X_test_final).T
Y_test_final = np.array(Y_test_final)
# print(X_test_final.shape, Y_test_final.shape)
# print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
X_centered = X_train - np.mean(X_train, axis=1, keepdims=True)
covariance = np.dot(X_centered, X_centered.T) / (X_train.shape[1] - 1)
eigenvalues, eigenvectors = np.linalg.eigh(covariance)
sorting = np.argsort(eigenvalues)[::-1]
U = eigenvectors[:, sorting]
U_p = U[:, :5]
Y_1 = U_p.T @ X_centered
X_test_reduced = U_p.T @ (X_test - np.mean(X_train, axis=1, keepdims=True))
X_test_final_reduced = U_p.T @ (X_test_final - np.mean(X_train, axis=1, keepdims=True))
# print(X_test_final_reduced.shape)
# print(Y_1.shape)
weights = np.array([1/len(Y_train) for i in range(len(Y_train))])
accuracy_list = []
stumps_list = []

for i in range(300):
    minimum_loss = 1
    optimal_split_dim = None
    optimal_split_value = None
    for j in range(Y_1.shape[0]):
        sorted_indices = np.argsort(Y_1[j])
        sorted_x = Y_1[j, sorted_indices]
        midpoints = (sorted_x[1:] + sorted_x[:-1]) / 2
        best_split = 0
        best_loss = 1
        for k in range(len(midpoints)):
            y_left_indices = sorted_indices[:k+1]
            y_right_indices = sorted_indices[k+1:]
            left_majority = 1 if np.sum(Y_train[y_left_indices]) > 0 else -1
            right_majority = 1 if np.sum(Y_train[y_right_indices]) > 0 else -1
            weights_left_indices = weights[y_left_indices]
            weights_right_indices = weights[y_right_indices]
            weights_left = np.sum(weights_left_indices[Y_train[y_left_indices] != left_majority])
            weights_right = np.sum(weights_right_indices[Y_train[y_right_indices] != right_majority])
            loss = (weights_left+weights_right)/np.sum(weights)
            if loss <= best_loss:
                best_loss = loss
                best_split = midpoints[k]
        if best_loss <= minimum_loss:
            minimum_loss = best_loss
            optimal_split_dim = j
            optimal_split_value = best_split
    sorted_indices = np.argsort(Y_1[optimal_split_dim])
    sorted_left_indices = []
    sorted_right_indices = []
    for j in range(len(sorted_indices)):
        if Y_1[optimal_split_dim][sorted_indices[j]] <= optimal_split_value:
            sorted_left_indices.append(sorted_indices[j])
        else:
            sorted_right_indices.append(sorted_indices[j])
    left_majority = 1 if np.sum(Y_train[sorted_left_indices]) > 0 else -1
    right_majority = 1 if np.sum(Y_train[sorted_right_indices]) > 0 else -1
    for j in range(len(sorted_left_indices)):
        if Y_train[sorted_left_indices[j]] != left_majority:
            weights[sorted_left_indices[j]] *= (1 - minimum_loss) / minimum_loss
    for j in range(len(sorted_right_indices)):
        if Y_train[sorted_right_indices[j]] != right_majority:
            weights[sorted_right_indices[j]] *= (1 - minimum_loss) / minimum_loss
    # print(weights[:10])
    print(optimal_split_dim, optimal_split_value, minimum_loss, i)
    alpha = np.log((1 - minimum_loss) / minimum_loss)
    stumps_list.append((optimal_split_dim, optimal_split_value, alpha))

for i in range(300):
    number_of_trees = i+1
    predictions = np.zeros(X_test_reduced.shape[1])
    for j in range(i+1):
        dim, value, alpha = stumps_list[j]
        predictions += alpha * np.sign(X_test_reduced[dim] - value)
    predictions_iteration = np.sign(predictions)
    val_accuracy = np.sum(predictions_iteration == Y_test) / len(Y_test)
    accuracy_list.append(val_accuracy)

print(accuracy_list)
plt.figure()
plt.plot(range(1, 301), accuracy_list, marker='o')
plt.xlabel('Number of Trees in ADA Boosting')
plt.ylabel('Val accuracy')
plt.title('ADA Boosting :Accuracy on Val Set vs. Number of Trees')
plt.grid(True)
plt.show()

max_accuracy = np.argmax(accuracy_list)
print("Best accuracy:", accuracy_list[max_accuracy])
predictions = np.zeros(X_test_final_reduced.shape[1])
for i in range(max_accuracy+1):
    dim, value, alpha = stumps_list[i]
    predictions += alpha * np.sign(X_test_final_reduced[dim] - value)
predictions_final = np.sign(predictions)
accuracy = np.sum(predictions_final == Y_test_final) / len(Y_test_final)
print("Accuracy for test set:", accuracy)
