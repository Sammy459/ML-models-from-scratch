import numpy as np
import matplotlib.pyplot as plt
import random as r


def qda(x, mean, covariance, inversed_covariance, prior):
    return ((-0.5) * np.dot(np.dot(x.T, inversed_covariance), x) + (np.dot((np.dot(inversed_covariance, mean)).T, x))
            - 0.5 * (np.dot(np.dot(mean.T, inversed_covariance), mean)) + np.log(prior))


dataset = np.load('mnist.npz')
x = dataset['x_train']
y = dataset['y_train']
x1 = dataset['x_test']
y1 = dataset['y_test']

classes = np.unique(y)
samples = 5

plt.figure(figsize=(10, 10))

for i, classLabel in enumerate(classes):
    classIndices = np.where(y == classLabel)[0]
    index = r.randint(0, len(classIndices) - 6)
    displaySamples = classIndices[index:index + 5]
    for j, index in enumerate(displaySamples):
        image = x[index]
        plt.subplot(len(classes), samples, i * samples + j + 1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
plt.show()

means = []
covariances = []
samples = []
x2 = x.reshape(x.shape[0], -1)
for i, classLabel in enumerate(classes):
    classIndices = np.where(y == classLabel)[0]
    X = []
    for j in range(len(classIndices)):
        X.append(x2[classIndices[j]])
    X= np.array(X).T
    samples.append(len(classIndices))
    mean = np.mean(X, axis=1, keepdims=True)
    means.append(mean)
    X_centered = X - np.mean(X, axis=1, keepdims=True)
    covariance = np.cov(X.T ,rowvar=False) + 0.01 * np.identity(X_centered.shape[0])
    covariances.append(covariance)

inversed_covariances = []
for i in range(len(covariances)):
    inversed_covariances.append(np.linalg.inv(covariances[i]))
priors = []
for i in range(len(classes)):
    priors.append(samples[i] / 60000)

x3 = x1.reshape(x1.shape[0], -1)
predictions = [0] * 10
total = 0
for i in range(len(x3)):
    prediction = []
    for j in range(len(classes)):
        prediction.append(qda(x3[i], means[j], covariances[j], inversed_covariances[j], priors[j]))
    predicted = np.argmax(prediction)
    if predicted == y1[i]:
        total += 1
        predictions[predicted] += 1
test_class_num = []
for i in range(10):
    test_class_num.append(len(np.where(y1 == i)[0]))
print("Overall accuracy: ", total / 10000)
for i in range(10):
    print("Accuracy for class ", i, " is ", predictions[i] / test_class_num[i])
