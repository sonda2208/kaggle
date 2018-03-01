# imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score
import sklearn.ensemble as ske

# preprocess data
def preprocess_data(data):
    # drop 'Ticket', 'Cabin', 'PassengerId' and 'Name'
    data = data.drop(['Ticket', 'Cabin', 'PassengerId', 'Name'], axis=1)

    # fill NaN value in 'Age' (TODO: apply predict model to fill this column
    mean_age = data['Age'].dropna().mean().astype(int)
    data['Age'] = data['Age'].fillna(mean_age)

    # add new feature 'FamilySize'
    data['FamilySize'] = data['SibSp'] + data['Parch']

    # drop 'SibSp' and 'Parch'
    data = data.drop(['SibSp', 'Parch'], axis=1)

    # fill NaN value in 'Embarked' with most frequent value
    most_fre_value = data.Embarked.dropna().mode()[0]
    data["Embarked"] = data["Embarked"].fillna(most_fre_value)

    # converting categorical feature to numeric
    data["Embarked"] = data["Embarked"].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    # convert categorical feature to numeric 
    data["Sex"] = data["Sex"].map( {'female': 1, 'male': 0} ).astype(int)

    # fill Nan value in 'Fare' with mean value
    mean_fare = data["Fare"].dropna().mean()
    data["Fare"] = data["Fare"].fillna(mean_fare)
    
    return data

### MAIN

# load train data
train_df = pd.read_csv("./train.csv")
train_df = preprocess_data(train_df)

# prepare data to fit model
X = train_df.drop(["Survived"], axis=1)
y = train_df["Survived"]

# apply random forest
rf = ske.RandomForestClassifier(n_estimators=100)
# scores = cross_val_score(rf, X, y, cv=10, scoring='accuracy')
# print(scores.mean())

# load test data
test_df = pd.read_csv("./test.csv")
passenger_id = test_df["PassengerId"]
test_df = preprocess_data(test_df)

print(test_df.head())

rf.fit(X, y)
y_pred = rf.predict(test_df)

submission = pd.DataFrame({
        "PassengerId": passenger_id,
        "Survived": y_pred
    })
submission.to_csv('./submission.csv', index=False)