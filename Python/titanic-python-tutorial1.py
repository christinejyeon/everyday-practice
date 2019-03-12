import pandas as pd
import numpy as np
import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv("/Users/Christine/Downloads/train.csv")
test_df = pd.read_csv("/Users/Christine/Downloads/test.csv")
combine = [train_df, test_df]

# describing the features of the data
print(train_df.columns.values)

# preview the data
print(train_df.head())

# figuring out the data type
print(train_df.info())
print(test_df.info())

# the dataset description
print(train_df.describe())
print(train_df.describe(include=['O']))

# analyze the survival rate by pivoting features
print(train_df[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train_df[['Sex','Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train_df[['SibSp','Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train_df[['Parch','Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# visualize the age values
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

# how's the class related to the survival rate?
# don't know why this is here, but
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();

# let's make graphs covering both Sex and Embarked features
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette = 'deep')
grid.add_legend()

# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette = {0:'k',1:'w'})
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()

# correcting: going to drop the features that we don't need for analysis
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
train_df = train_df.drop(['Ticket', 'Cabin'], axis = 1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis = 1)
combine = [train_df, test_df]
print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

# going to create new feature extracting from existing
# the reason for doing this: because there was an assumption above
# regarding that most titles band age group, and
# some titles definitely survived while other titles did not
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand = False)
print(pd.crosstab(train_df['Title'], train_df['Sex']))

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
        'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] = dataset['Title'].replace('Ms','Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
print(train_df[['Title','Survived']].groupby(['Title'], as_index=False).mean())

title_mapping = {"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Rare":5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
print(train_df.head())

# Now we can drop the 'Name' feature!
train_df = train_df.drop(['Name','PassengerId'], axis = 1)
test_df = test_df.drop(['Name'], axis = 1)
combine = [train_df, test_df]
print(train_df.shape, test_df.shape)

# convert strings to numerical values
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female':1, 'male':0}).astype(int)
print(train_df.head())

# filling in the missing or null values, first with the Age feature
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=0.6)
grid.map(plt.hist, 'Age', alpha=.5, bins = 20)
grid.add_legend()
# going to start guessing ages, starting from making an array
guess_ages = np.zeros((2,3))
for dataset in combine:
    for i in range(0,2):
        for j in range(0,3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            guess_ages[i,j] = int( age_guess/0.5 + 0.5) * 0.5

    for i in range(0,2):
        for j in range(0,3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1), \
                        'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)
print(train_df.head())
# now create age bands, and determine correlations with Survived
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))
# replace Age with ordinals based on these bands.
for dataset in combine:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[ (dataset['Age'] > 16) & (dataset['Age'] <=32), 'Age'] = 1
    dataset.loc[ (dataset['Age'] > 32) & (dataset['Age'] <=48), 'Age'] = 2
    dataset.loc[ (dataset['Age'] > 48) & (dataset['Age'] <=64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
print(train_df.head())
# drop the AgeBand feature
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
print(train_df.head())

# Now combine a new feature for FamilySize
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
print(train_df.head())
print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# and for the people who are alone
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
print(train_df[['IsAlone','Survived']].groupby(['IsAlone'], as_index=False).mean())
print(train_df.head())
# as we've made the feature IsAlone, we're going to drop the features that are useless from now on
train_df = train_df.drop(['SibSp','Parch','FamilySize'], axis=1)
test_df = test_df.drop(['SibSp', 'Parch', 'FamilySize'], axis=1)
combine = [train_df, test_df]
print(train_df.head())

# Now create an artificial feature combining Pclass and Age
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass
print(train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10))

# Embarked feature will be only categorized into what's frequent or # NOTE:
freq_port = train_df.Embarked.dropna().mode()[0]
print(freq_port)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
print(train_df[['Embarked','Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# combine categorical feature into numeric
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S':0, 'C':1, 'Q':2}).astype(int)
print(train_df.head())

# We're going to put mode to the unknown value in the Fare feature
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
print(test_df.head())
# ********* I don't get the process here *********
# Why are we using the mode value in a test dataset?
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
print(train_df[['FareBand','Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))
# convert the Fare feature to ordinal values based on the FareBand
for dataset in combine:
    dataset.loc[ dataset['Fare'] <=7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <=14.454),'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <=31), 'Fare'] = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

print(train_df.head(10))
print(test_df.head(10))


# Now we are ready to train a model and predict the required solution!
# Our problem is a classification and regression problem
# Need to review all the data mining methods that I've learned

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
# ********** Why should I divide the two? We'll figure it out
X_test = test_df.drop("PassengerId", axis=1).copy()
print(X_train.shape, Y_train.shape, X_test.shape)

# Logistic regression is a useful model to run early in the workflow
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print("Logistic Regression: ", acc_log)
# by calculating the coefficient of the features,
# we can use logistic regression to validate our assumptions and decisions
# Positive coefficients increase the log-odds of the response
# and increase the probability
# Negative coefficients on the opposite
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
print(coeff_df.sort_values(by="Correlation", ascending=False))

# Now we model using SVM. It generates a confidence score
# which is higher than Logistics Regression model
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
print("SVC: ", acc_svc)

# k-NN
# confidence score is better than Logistics Regression
# but worse than SVM
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print("k-NN: ", acc_knn)

# Gaussian Naive Bayes
# confidence score the lowest among the ones done til now
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gau = round(gaussian.score(X_train, Y_train) * 100, 2)
print("Gaussian Naive Bayes: ", acc_gau)

# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
print("Perceptron: ", acc_perceptron)

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linearsvc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print("Linear SVC: ", acc_linearsvc)

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
print("Stochastic Gradient Descent: ", acc_sgd)

# Decision tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decisiontree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print("Decision Tree: ", acc_decisiontree)

# Random Forest
# Ensemble learning method for classification, regression, etc
# Operate a multitude of decision trees
# The model confidence score is the highest
random_forest = RandomForestClassifier()
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
acc_rf = round(random_forest.score(X_train, Y_train) * 100, 2)
print("Random Forest: ", acc_rf)

# Model Evaluation
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
                'Random Forest', 'Naive Bayes', 'Perceptron',
                'Stochastic Gradient Decent', 'Linear SVC',
                'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log,
               acc_rf, acc_gau, acc_perceptron,
               acc_sgd, acc_linearsvc, acc_decisiontree]})
print(models.sort_values(by='Score', ascending=False))

submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": decision_tree.predict(X_test)
})
