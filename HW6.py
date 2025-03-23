# HW6 - Spam Classification
# implement a Naive Bayes classifier to classify spam and non-spam emails

# Each feature vector is 344 characters long, where each character represent a unique word that would potentially identify e-mail as spam or non-spam. The value of each character is a boolean value that identifies weather or not the word has one or more occurrences in the e-mail (value 1) and value 0 if the word does not appear in the document. 
# First column is a vector instance number and the second column identifies if the vector represents a spam e-mail message (value 1) or non-spam message (value -1)

import numpy as np

data = np.loadtxt('SpamInstances.txt', delimiter=' ', skiprows=1)
print(data.shape)
print(data.shape[0])

# Split the data into training and testing sets
np.random.shuffle(data)
train_data = data[:int(0.8*data.shape[0])]
test_data = data[int(0.8*data.shape[0]):]

# get labels
train_labels = test_data[:, 1]
test_labels = test_data[:, 1]

# get vector instance numbers
train_instance = train_data[:, 0]
test_instance = test_data[:, 0]

# get the vector instances
train_data = train_data[:, 2]
test_data = test_data[:, 2]

# Calculate the prior probabilities
spam_count = 0
non_spam_count = 0
for i in range(train_labels.shape[0]):
    if train_labels[i] == 1:
        spam_count += 1
    else:
        non_spam_count += 1

prior_spam = spam_count / train_data.shape[0]
prior_non_spam = non_spam_count / train_data.shape[0]

# Calculate the conditional probabilities
spam_words = np.zeros(57)
non_spam_words = np.zeros(57)
for i in range(train_data.shape[0]):
    if train_data[i, 57] == 1:
        spam_words += train_data[i, :57]
    else:
        non_spam_words += train_data[i, :57]

spam_words = spam_words / spam_count
non_spam_words = non_spam_words / non_spam_count

# Classify the test data
correct = 0
for i in range(test_data.shape[0]):
    spam_prob = prior_spam
    non_spam_prob = prior_non_spam
    for j in range(57):
        if test_data[i, j] == 1:
            spam_prob *= spam_words[j]
            non_spam_prob *= non_spam_words[j]
        else:
            spam_prob *= (1 - spam_words[j])
            non_spam_prob *= (1 - non_spam_words[j])

    if spam_prob > non_spam_prob:
        prediction = 1
    else:
        prediction = 0

    if prediction == test_data[i, 57]:
        correct += 1

accuracy = correct / test_data.shape[0]
print("Accuracy: ", accuracy)



