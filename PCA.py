import matplotlib.pyplot as plt
import numpy as np
import random as r

def qda(x, mean, inversed_covariance, prior):
    return ((-0.5) * np.dot(np.dot(x.T, inversed_covariance), x) + (np.dot((np.dot(inversed_covariance, mean)).T, x))
            - 0.5 * (np.dot(np.dot(mean.T, inversed_covariance), mean)) + np.log(prior))


data = np.load("mnist.npz")
x = data['x_train']
y = data['y_train']
x1 = data['x_test']
y1 = data['y_test']
samples = 100
x2 = x.reshape(x.shape[0], -1)
classes = np.unique(y)
array1 = []
for i, classlabel in enumerate(classes):
    classIndices = np.where(y == classlabel)[0]
    displaySamples = classIndices[:100]
    for j in range(100):
        array1.append(x2[displaySamples[j]])
array2 = np.array(array1)
X = array2.T
X_centered = X - np.mean(X, axis=1, keepdims=True)
covariance = np.dot(X_centered, X_centered.T) / 999
eigenvalues, eigenvectors = np.linalg.eigh(covariance)
sorting = np.argsort(eigenvalues)[::-1]
U = eigenvectors[:, sorting]
Y = np.dot(U.T, X_centered)
X_new = np.dot(U, Y)
MSE = 0
for index in range(784):
    for j in range(1000):
        MSE += (X_centered[index][j] - X_new[index][j]) ** 2
print("The MSE is:", MSE)

to_use = [5, 10, 20]
for index in to_use:
    U_p = U[:, :index]
    Y_1 = U_p.T @ X_centered
    new = U_p @ Y_1 + np.mean(X, axis=1, keepdims=True)
    new = new.reshape(28, 28, -1)
    plt.figure(figsize=(10, 10))
    for j in range(10):
        for k in range(5):
            image = new[:, :, j * 100 + k]
            plt.subplot(10, 5, j * 5 + k + 1)
            plt.axis('off')
            plt.imshow(image, cmap='gray')
    plt.show()

x3 = x1.reshape(x1.shape[0], -1)
x3 = x3.T

to_use = [5, 10, 20]
for index in to_use:
    U_p = U[:, :index]
    Y_1 = U_p.T @ X_centered
    means = []
    inversed_covariances = []
    predictions = [0] * 10
    total = 0
    for j in range(10):
        temp1 = Y_1[:, j * 100:j * 100 + 100]
        means.append(np.mean(temp1, axis=1, keepdims=True))
        inversed_covariances.append(np.linalg.inv(np.cov(temp1.T, rowvar=False) + 0.0001 * np.identity(temp1.shape[0])))
    U_test_P = U[:, :index]
    X_center = x3 - np.mean(x3, axis=1, keepdims=True)
    Y_test = U_test_P.T @ X_center
    for j in range(10000):
        prediction = []
        temp1 = Y_test[:, j]
        for k in range(10):
            prediction.append(
                qda(temp1, means[k], inversed_covariances[k], 0.1))
        predicted = np.argmax(prediction)
        if predicted == y1[j]:
            predictions[predicted] += 1
            total += 1
    print("p =", index)
    test_class_num = []
    for i in range(10):
        test_class_num.append(len(np.where(y1 == i)[0]))
    print("Overall accuracy: ", total / 10000)
    for i in range(10):
        print("Accuracy for class ", i, " is ", predictions[i] / test_class_num[i])


