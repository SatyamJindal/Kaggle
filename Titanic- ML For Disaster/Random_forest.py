'''

Accuracy - 0.74641

'''

import pandas as pd
from sklearn.tree import DecisionTreeClassifier as df
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.metrics import mean_absolute_error as mae

df_train = pd.read_csv("C:/Users/satyam/Desktop/kaggle/new_titanic/train.csv")
df_test = pd.read_csv("C:/Users/satyam/Desktop/kaggle/new_titanic/test.csv")



survived_train = df_train.Survived 
data = pd.concat([df_train.drop(['Survived'],axis=1),df_test]) 
data['Age'] = data.Age.fillna(data.Age.mean())
data['Fare'] = data.Fare.fillna(data.Fare.mean()) # Filling missing values
#Now we will change our values to numbers using get_dummies
data = pd.get_dummies(data, columns=['Sex'],drop_first=True) # Creates two columns each for male and female, drop_first is used as just the male can give you info about both
data = data[['Sex_male','Fare','Age','Pclass','SibSp']] # Dataset with only these columns
data_train = data.iloc[:891]
data_test = data.iloc[891:] # Back to splitting both
#Scikit - learn requires data as arrays ans not data frames so we convert data into arrays
x = data_train.values
test = data_test.values
y = survived_train.values
clf = rf() # Defining the Random Forest
clf.fit(x,y) # features + target
y_pred = clf.predict(test)
df_test['Survived'] = y_pred
df_test['Survived'] = df_test.Survived.apply(lambda x: int(x))
df_test[['PassengerId','Survived']].to_csv('C:/Users/satyam/Desktop/kaggle/new_titanic/my_model_rf.csv',index=False)

