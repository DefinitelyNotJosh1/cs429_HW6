# HW6 - Spam Classification
# implement a Naive Bayes classifier to classify spam and non-spam emails

# Each feature vector is 344 characters long, where each character represent a unique word that would potentially identify e-mail as spam or non-spam. The value of each character is a boolean value that identifies weather or not the word has one or more occurrences in the e-mail (value 1) and value 0 if the word does not appear in the document. 
# First column is a vector instance number and the second column identifies if the vector represents a spam e-mail message (value 1) or non-spam message (value -1)

import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        self.prior = None
        self.likelihood = None
        self.posterior = None
    
    def train(self, data, labels):
        # get the number of spam and non-spam emails
        spam = np.sum(labels == 1)
        non_spam = np.sum(labels == -1)
        
        # get the prior probability
        self.prior = spam / (spam + non_spam)
        
        # get the likelihood
        self.likelihood = np.zeros((2, data.shape[1]))
        for i in range(data.shape[1]):
            self.likelihood[0, i] = np.sum(data[labels == 1, i]) / spam
            self.likelihood[1, i] = np.sum(data[labels == -1, i]) / non_spam

    def predict(self, data):
        self.posterior = np.zeros((data.shape[0], 2))
        for i in range(data.shape[0]):
            self.posterior[i, 0] = np.prod(data[i] * self.likelihood[0]) * self.prior
            self.posterior[i, 1] = np.prod(data[i] * self.likelihood[1]) * (1 - self.prior)

        return np.argmax(self.posterior, axis=1)
    
    def accuracy(self, labels):
        return np.sum(labels == self.predicted_labels) / labels.shape[0]
    


# Load the data
data = np.loadtxt('SpamInstances.txt', delimiter=' ', skiprows=1)

print(data.shape)

# Split the data into training and testing sets
start = int(0.1*data.shape[0])
end = int(0.9*data.shape[0])
middle = int(0.5*data.shape[0])

train_data_ham = data[start:end]
train_data_spam = data[middle:end]

valid_data_ham = data[:start]
valid_data_spam = data[end:]

# get the feature vectors
vector_data = np.zeros((data.shape[0] - 1, 334))

print(vector_data.shape)

# function to process each line
def process_line(line):
    # split the line into parts
    parts = line.split()
    # print(parts)
    instance = int(parts[0])
    label = int(parts[1])
    binary_str = parts[2]
    
    # convert binary string to numpy array of digits
    binary_array = np.array([int(d) for d in binary_str])
    
    return instance, label, binary_array

with open('SpamInstances.txt',) as f:
    for i, line in enumerate(f):
        if i == 0:
            continue
        instance, label, binary_array = process_line(line)
        vector_data[i-1] = binary_array
        

print(vector_data.shape)





