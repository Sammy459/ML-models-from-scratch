import numpy as np
import matplotlib.pyplot as plt


def spiltting_datset(X, split_dim, split_value):
    left_indices = []
    right_indices = []
    for i in range(X.shape[1]):
        if X[split_dim, i] <= split_value:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return left_indices, right_indices


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
X_train = np.array(array1).T
Y_train = np.array(array2)
X_test = np.array(array3).T
Y_test = np.array(array4)

X_test_final = []
Y_test_final = []
for i in range(len(y1)):
    if y1[i] == 0:
        X_test_final.append(x3[i])
        Y_test_final.append(-1)
    elif y1[i] == 1:
        X_test_final.append(x3[i])
        Y_test_final.append(1)
X_test_final = np.array(X_test_final).T
Y_test_final = np.array(Y_test_final)

X_centered = X_train - np.mean(X_train, axis=1, keepdims=True)
covariance = np.dot(X_centered, X_centered.T) / (X_train.shape[1] - 1)
eigenvalues, eigenvectors = np.linalg.eigh(covariance)
sorting = np.argsort(eigenvalues)[::-1]
U = eigenvectors[:, sorting]
U_p = U[:, :5]
X_train_reconstructed = U_p.T @ X_centered
X_test_reconstructed = U_p.T @ (X_test - np.mean(X_train, axis=1, keepdims=True))
X_test_final_reconstructed = U_p.T @ (X_test_final - np.mean(X_train, axis=1, keepdims=True))

mse_lists = []
stump_lists = []
residuals = Y_train.astype(float)

for i in range(300):
    best_dim = 0
    best_ssr = float('inf')
    best_left_mean = 0
    best_right_mean = 0
    best_split = 0
    for j in range(X_train_reconstructed.shape[0]):
        sorted_indices = np.argsort(X_train_reconstructed[j])
        sorted_x = X_train_reconstructed[j, sorted_indices]
        sorted_y = Y_train[sorted_indices]
        midpoints = (sorted_x[1:] + sorted_x[:-1]) / 2
        for k in range(len(midpoints)):
            split = midpoints[k]
            left_indices = sorted_indices[:k+1]
            right_indices = sorted_indices[k+1:]
            left_mean = np.mean(Y_train[left_indices])
            right_mean = np.mean(Y_train[right_indices])
            ssr = np.sum((Y_train[left_indices] - left_mean)**2) + np.sum((Y_train[right_indices] - right_mean)**2)
            if ssr < best_ssr:
                best_ssr = ssr
                best_dim = j
                best_left_mean = left_mean
                best_right_mean = right_mean
                best_split = split
    sorted_indices = np.argsort(X_train_reconstructed[best_dim])
    residues_best = residuals[sorted_indices]
    best_sorted_x = X_train_reconstructed[best_dim, sorted_indices]
    best_sorted_y = Y_train[sorted_indices]
    left_y = best_sorted_y[best_sorted_x < best_split]
    right_y = best_sorted_y[best_sorted_x > best_split]
    best_left_mean = np.mean(left_y)
    best_right_mean = np.mean(right_y)
    residues_best[best_sorted_x < best_split] -= 0.01*best_left_mean
    residues_best[best_sorted_x > best_split] -= 0.01*best_right_mean
    y = np.sign(residues_best)
    original = np.argsort(sorted_indices)
    Y_train = y[original]
    residuals = residues_best[original]
    decision_left = np.mean(y[best_sorted_x < best_split])
    decision_right = np.mean(y[best_sorted_x > best_split])
    stump_lists.append((best_dim, best_split, decision_left, decision_right, best_ssr))
    predictions = np.zeros(len(Y_test))
    for j in stump_lists:
        dim, split, left, right, error = j
        predictions += np.where(X_test_reconstructed[dim] < split, left, right)*0.01
    mse = np.mean((predictions-Y_test) ** 2)
    mse_lists.append(mse)
    print(mse, i)

best_ssr = np.argmin(mse_lists)
print("Lowest MSE:", mse_lists[best_ssr])
print("Best tree:", best_ssr+1)

plt.figure()
plt.plot(range(1, 301), mse_lists, marker='o')
plt.xlabel('Number of Trees in Gradient Boosting Regression')
plt.ylabel('MSE on Validation Set')
plt.title('Gradient Boosting Regression: MSE on Validation Set vs. Number of Trees')
plt.grid(True)
plt.show()

predictions = np.zeros(len(Y_test_final))
for i in range(best_ssr):
    dim, split, left_mean, right_mean, error = stump_lists[i]
    predictions += np.where(X_test_final_reconstructed[dim] < split, left_mean, right_mean)*0.01
mse = np.mean((predictions - Y_test_final) ** 2)
print("MSE on test set:", mse)
