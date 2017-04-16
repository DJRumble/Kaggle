import pandas
# Import the linear regression class
from sklearn.linear_model import LinearRegression
# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.cross_validation import KFold
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation


titanic_test = pandas.read_csv("test.csv")
titanic_train = pandas.read_csv("train.csv")

###### FORMAT TEST DATA FOR TESTING PURPOSES ######
#fill null ages with median
titanic_test["Age"] = titanic_test["Age"].fillna(titanic_test["Age"].median())
#fill null fares with median
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
#numurate Sex
titanic_test.loc[titanic_test["Sex"] == "male","Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female","Sex"] = 1

#numerate Embarked
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")
titanic_test.loc[titanic_test["Embarked"] == "S","Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C","Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q","Embarked"] = 2

###### FORMAT TRAIN DATA FOR MACHINE LEARNING PURPOSES ######
#fill null ages with median
titanic_train["Age"] = titanic_train["Age"].fillna(titanic_train["Age"].median())
#fill null fares with median
titanic_train["Fare"] = titanic_train["Fare"].fillna(titanic_train["Fare"].median())
#numurate Sex
titanic_train.loc[titanic_train["Sex"] == "male","Sex"] = 0
titanic_train.loc[titanic_train["Sex"] == "female","Sex"] = 1

#numerate Embarked
titanic_train["Embarked"] = titanic_train["Embarked"].fillna("S")
titanic_train.loc[titanic_train["Embarked"] == "S","Embarked"] = 0
titanic_train.loc[titanic_train["Embarked"] == "C","Embarked"] = 1
titanic_train.loc[titanic_train["Embarked"] == "Q","Embarked"] = 2

######## DEFFINE PREDICTORS #####
# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

######### LLN REGRESS. ########
print("=========")
print("LIN. REGS")
print("=========")

alg = LinearRegression()

kf = KFold(titanic_train.shape[0], n_folds=3, random_state=1)

predictions1 = []
for train, test in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (titanic_train[predictors].iloc[train,:])
    # The target we're using to train the algorithm.
    train_target = titanic_train["Survived"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(titanic_train[predictors].iloc[test,:])
    predictions1.append(test_predictions)

#print(predictions)

predictions1 = np.concatenate(predictions1, axis=0)

# Map predictions to outcomes (only possible outcomes are 1 and 0)
predictions1[predictions1 > .5] = 1
predictions1[predictions1 <=.5] = 0

accuracy = (sum(predictions1[predictions1 == titanic_train["Survived"]])/len(predictions1))

submission = pandas.DataFrame({
        "PassengerId": titanic_train["PassengerId"],
        "Survived": predictions1
    })

submission.to_csv("kaggle_linreg.csv", index=False)

print(accuracy)
####### LOG REGRESS. ##########
print("=========")
print("LOG. REGS")
print("=========")

predictions2 = []
# Initialize the algorithm class
alg = LogisticRegression(random_state=1)
scores = cross_validation.cross_val_score(alg, titanic_train[predictors], titanic_train["Survived"], cv=3)
#print(scores)
# Train the algorithm using all the training data
alg.fit(titanic_train[predictors], titanic_train["Survived"])
# Make predictions using the test set.
predictions2 = alg.predict(titanic_test[predictors])

# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pandas.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions2
    })

submission.to_csv("kaggle_logreg.csv", index=False)
