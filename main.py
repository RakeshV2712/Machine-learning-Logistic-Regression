# Importing dependencies

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Data Collection and Processing

sonar_data = pd.read_csv('C:\Projects\Simple machine learning classification\Copy of sonar data.csv', header=None)

# Prints first 5 values in the csv file

print(sonar_data.head())

# Prints the dimensions of the CSV

print(sonar_data.shape)

# Lists the basic statistical description of the data

print((sonar_data.describe()))

# get the distribution of mines (M) vs rocks (R) in the data

print(sonar_data[60].value_counts())

# Mean of the data

print(sonar_data.groupby(60).mean())

# Separating the columns into data and labels

X = sonar_data.drop(columns=60,axis=1)
Y = sonar_data[60]

print(X)
print(Y)

# Separating the training and test data

X_train , X_test, Y_train, Y_test = train_test_split(X ,Y, test_size = 0.1, stratify= Y, random_state=1)

# Training the model -----> logistic regression model

model = LogisticRegression()

model.fit(X_train, Y_train)

# model evaluation

# 1 Accuracy on training data

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print(training_data_accuracy)

# 2 Accuracy on test data

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)

print(test_data_accuracy)

# Making a predictive system

input_data = (0.0164,0.0173,0.0347,0.0070,0.0187,0.0671,0.1056,0.0697,0.0962,0.0251,0.0801,0.1056,0.1266,0.0890,0.0198,0.1133,0.2826,0.3234,0.3238,0.4333,0.6068,0.7652,0.9203,0.9719,0.9207,0.7545,0.8289,0.8907,0.7309,0.6896,0.5829,0.4935,0.3101,0.0306,0.0244,0.1108,0.1594,0.1371,0.0696,0.0452,0.0620,0.1421,0.1597,0.1384,0.0372,0.0688,0.0867,0.0513,0.0092,0.0198,0.0118,0.0090,0.0223,0.0179,0.0084,0.0068,0.0032,0.0035,0.0056,0.0040)

input_data_as_numpy = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy.reshape(1,-1)

prediction = model.predict(input_data_reshaped)

print(prediction)

if prediction[0] == 'R':

    print("The object is a Rock")

else:
    print("The object is a Mine")