# HW6 - Spam Classification
# Author - Joshua Krasnogorov
# Due date - 4/6/2025
# implement a Naive Bayes classifier to classify spam and non-spam emails

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NaiveBayesClassifier:
    def __init__(self):
        self.prior = None
        self.likelihood = None
        self.posterior = None
        self.predicted_labels = None
    
    def train(self, data, labels):
        spam = np.sum(labels == 1)
        non_spam = np.sum(labels == -1)
        
        # prior probability
        self.prior = spam / (spam + non_spam)
        
        # likelihood with Laplace smoothing
        self.likelihood = np.zeros((2, data.shape[1]))
        for i in range(data.shape[1]):
            self.likelihood[0, i] = (np.sum(data[labels == 1, i]) + 1) / (spam + 2)
            self.likelihood[1, i] = (np.sum(data[labels == -1, i]) + 1) / (non_spam + 2)

    def predict(self, data):
        self.posterior = np.zeros((data.shape[0], 2))
        for i in range(data.shape[0]):

            # log likelihood for spam
            log_likelihood_spam = np.sum(data[i] * np.log(self.likelihood[0]) + (1 - data[i]) * np.log(1 - self.likelihood[0]))

            # log likelihood for non-spam
            log_likelihood_nonspam = np.sum(data[i] * np.log(self.likelihood[1]) + (1 - data[i]) * np.log(1 - self.likelihood[1]))

            # log posterior probabilities
            self.posterior[i, 0] = log_likelihood_spam + np.log(self.prior)
            self.posterior[i, 1] = log_likelihood_nonspam + np.log(1 - self.prior)
        
        self.predicted_labels = np.where(np.argmax(self.posterior, axis=1) == 0, 1, -1)
        return self.predicted_labels
    
    def accuracy(self, labels):
        # calculate accuracy as proportion of correct predictions
        return np.sum(labels == self.predicted_labels) / labels.shape[0]

# use pandas to read the instance vector as a string for processing
df = pd.read_csv("SpamInstances.txt", delimiter=' ', skip_blank_lines=True, skiprows=1, header=0, dtype=str)
df.columns = ['instance', 'label', 'vectors']
df = df.dropna()

# convert binary string column to numpy arrays
vector_data = np.array([np.array([int(d) for d in binary]) for binary in df['vectors']])
labels = df['label'].astype(int).to_numpy()

# split spam and non-spam instances
spam_indices = np.where(labels == 1)[0]
non_spam_indices = np.where(labels == -1)[0]

# prepare for 20 iterations
n_iterations = 20
train_sizes = np.linspace(100, 12400, n_iterations, dtype=int)
precisions = []
recalls = []
accuracies = []
false_positives = []
false_negatives = []

# run 20 iterations
for train_size in train_sizes:
    # ensure equal spam/non-spam split
    half_size = train_size // 2
    
    # randomly sample from spam and non-spam
    np.random.shuffle(spam_indices)
    np.random.shuffle(non_spam_indices)
    
    train_spam_idx = spam_indices[:half_size]
    train_nonspam_idx = non_spam_indices[:half_size]
    train_idx = np.concatenate([train_spam_idx, train_nonspam_idx])
    
    # test set is remaining instances
    test_idx = np.setdiff1d(np.arange(len(labels)), train_idx)
    
    # create train and test sets
    train_data = vector_data[train_idx]
    train_labels = labels[train_idx]
    test_data = vector_data[test_idx]
    test_labels = labels[test_idx]
    
    # train and predict
    classifier = NaiveBayesClassifier()
    classifier.train(train_data, train_labels)
    predictions = classifier.predict(test_data)
    
    # calculate metrics
    true_pos = np.sum((predictions == 1) & (test_labels == 1))
    false_pos = np.sum((predictions == 1) & (test_labels == -1))
    false_neg = np.sum((predictions == -1) & (test_labels == 1))
    actual_pos = np.sum(test_labels == 1)
    
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    accuracy = classifier.accuracy(test_labels)
    
    precisions.append(precision)
    recalls.append(recall)
    accuracies.append(accuracy)
    false_positives.append(false_pos)
    false_negatives.append(false_neg)

# plotting
plt.figure(figsize=(12, 6))
plt.plot(train_sizes, precisions, 'b-', label='Precision')
plt.plot(train_sizes, recalls, 'r-', label='Recall')
plt.plot(train_sizes, accuracies, 'g-', label='Accuracy')
plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.title('Classifier Performance vs Training Size')
plt.legend()
plt.grid(True)
plt.savefig("classifier_performance.png")
plt.show()

# print results
for i, size in enumerate(train_sizes):
    print(f"Train size: {size}")
    print(f"Precision: {precisions[i]:.4f}")
    print(f"Recall: {recalls[i]:.4f}")
    print(f"Accuracy: {accuracies[i]:.4f}")
    print(f"False Positives: {false_positives[i]}")
    print(f"False Negatives: {false_negatives[i]}\n")