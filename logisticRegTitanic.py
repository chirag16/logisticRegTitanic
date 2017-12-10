import numpy as np
import pandas as pd

# getting the data
data = pd.read_csv('train.csv')
data.drop(data.columns[[0, 3, 7, 8, 9, 10, 11]], axis=1, inplace=True)

print "DATA\n"
print data.head()

X = data.drop(data.columns[[0]], axis=1, inplace=False)
y = data[['Survived']]

# cleaning the data
sex = X[['Sex']].values
sex[sex == 'male'] = 0
sex[sex == 'female'] = 1
X[['Sex']] = sex

age = X[['Age']].values
max_age = max(age)
age /= max_age
X[['Age']] = age

# deleting entries with NaN ages
to_be_deleted = []
for i in range(len(age)):
	if np.isnan(age[i]):
		to_be_deleted.append(i)
X.drop(X.index[to_be_deleted], axis=0, inplace=True)
y.drop(y.index[to_be_deleted], axis=0, inplace=True)

pclass = X[['Pclass']].values.astype(float)
max_pclass = max(pclass)
pclass /= max_pclass
X[['Pclass']] = pclass

sibsp = X[['SibSp']].values.astype(float)
max_sibsp = max(sibsp)
sibsp /= max_sibsp
X[['SibSp']] = sibsp

print X.head()
print "\n"

X = X.as_matrix()
y = y.as_matrix()

def predict(x):
	global theta

	h = np.dot(theta, x)
	h = 1. / (1 + np.exp(-h))

	return h

# Regression line

alpha = 0.01  # learning rate
theta = np.array([0 for i in range(X.shape[1] + 1)]).astype(float)
num_epochs = 50
# traning
print "TRAINING",
for n in range(num_epochs):
	for i in range(X.shape[0]):
		x = [1]
		for element in X[i]:
			x.append(element)
		h = predict(x)

		# grad ascent
		for j in range(len(theta)):
			theta[j] += alpha * (y[i] - h) * x[j]
	print ".",

# predictions
data_test = pd.read_csv('test.csv')
labels = pd.read_csv('gender_submission.csv')

data_test.drop(data_test.columns[[0, 2, 6, 7, 8, 9, 10]], axis=1, inplace=True)

X_test = data_test
y_test = labels[['Survived']]

# cleaning the data
sex = X_test[['Sex']].values
sex[sex == 'male'] = 0
sex[sex == 'female'] = 1
X_test[['Sex']] = sex

age = X_test[['Age']].values
max_age = max(age)
age /= max_age
X_test[['Age']] = age

# deleting entries with NaN ages
to_be_deleted = []
for i in range(len(age)):
	if np.isnan(age[i]):
		to_be_deleted.append(i)
X_test.drop(X_test.index[to_be_deleted], axis=0, inplace=True)
y_test.drop(y_test.index[to_be_deleted], axis=0, inplace=True)

pclass = X_test[['Pclass']].values.astype(float)
max_pclass = max(pclass)
pclass /= max_pclass
X_test[['Pclass']] = pclass

sibsp = X_test[['SibSp']].values.astype(float)
max_sibsp = max(sibsp)
sibsp /= max_sibsp
X_test[['SibSp']] = sibsp

X_test = X_test.as_matrix()
y_test = y_test.as_matrix()

print "\nRESULTS"

error = 0.
for i in range(10):
	x = [1]
	for element in X_test[i]:
		x.append(element)
	if predict(x) > 0.5:
		pred = 1
	else: pred = 0
	print pred, "\t" + str(y_test[i][0])

	if pred != y_test[i][0]:
		error += 1

print "\nACCURACY", (10 - error) / 10