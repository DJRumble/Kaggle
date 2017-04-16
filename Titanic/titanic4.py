import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import Machine Learning libaries
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier



def df_cleaner(df):
    """
    Clean up a few variables in the training/test sets.
    """
    
    # Clean up ages.
    for passenger in df[(df['Age'].isnull())].index:
        df.loc[passenger, 'Age'] = np.average(df[(df['Age'].notnull())]['Age'])

    # Clean up fares.
    for passenger in df[(df['Fare'].isnull())].index:
        df.loc[passenger, 'Fare'] = np.average(df[(df['Fare'].notnull())]['Fare'])

    # Manually convert values to numeric columns for clarity.
    # Change the sex to a binary column.
    df['Sex'][(df['Sex'] == 'male')] = 0
    df['Sex'][(df['Sex'] == 'female')] = 1
    df['Sex'][(df['Sex'].isnull())] = 2

    # Transform to categorical data.
    df['Embarked'][(df['Embarked'] == 'S')] = 0
    df['Embarked'][(df['Embarked'] == 'C')] = 1
    df['Embarked'][(df['Embarked'] == 'Q')] = 2
    df['Embarked'][(df['Embarked'].isnull())] = 3

    return df

#Setting up DATA
titanic_test = df_cleaner(pd.read_csv("test.csv"))
titanic = df_cleaner(pd.read_csv("train.csv"))

    # Remove unused columns, clean age, and convert gender to binary column.
titanic_test.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
    # Remove unused columns, clean age, and convert gender to binary column.
titanic.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)

##### MACHINE LEARNING ######

predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked"]#, "FamilySize", "Title", "FamilyId"]

algorithms = [[GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],[LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]]

titanic_test[predictors].astype(float).to_csv("predictors.csv",index=False)

full_predictions = []
for alg, predictors in algorithms:
    # Fit the algorithm using the full training data.
    alg.fit(titanic[predictors], titanic["Survived"])
    # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
    predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:,1]
    full_predictions.append(predictions)	

# The gradient boosting classifier generates better predictions, so we weight it higher.
predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4

#print(predictions)
predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1

predictions = predictions.astype(int)

submission = pandas.DataFrame({"PassengerId": titanic_test["PassengerId"],"Survived":predictions})

submission.to_csv("kaggle4.csv",index=False)
