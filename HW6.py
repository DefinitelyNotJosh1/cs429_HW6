# HW6 - Spam Classification
# implement a Naive Bayes classifier to classify spam and non-spam emails

# Each feature vector is 344 characters long, where each character represent a unique word that would potentially identify e-mail as spam or non-spam. The value of each character is a boolean value that identifies weather or not the word has one or more occurrences in the e-mail (value 1) and value 0 if the word does not appear in the document. 
# First column is a vector instance number and the second column identifies if the vector represents a spam e-mail message (value 1) or non-spam message (value -1)

import numpy as np

data = np.loadtxt('SpamInstances.txt', delimiter=' ', skiprows=1)

# Split the data into training and testing sets
np.random.shuffle(data)
start = int(0.1*data.shape[0])
end = int(0.9*data.shape[0])

train_data = data[start:end]
test_data = data[-start:start]

print(train_data.shape)
print(test_data.shape)

# get labels
train_labels = test_data[:, 1]
test_labels = test_data[:, 1]

# get vector instance numbers
train_instance = train_data[:, 0]
test_instance = test_data[:, 0]

# get the vector instances
train_data = train_data[:, 2]
test_data = test_data[:, 2]








