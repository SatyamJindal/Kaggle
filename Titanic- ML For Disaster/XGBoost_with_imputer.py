'''

Accuracy -  76.555

'''


import pandas as pd
from sklearn.tree import DecisionTreeClassifier as df
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.metrics import mean_absolute_error as mae                                        #Importing Libraries
from sklearn.preprocessing import Imputer
from xgboost import XGBClassifier

df_train = pd.read_csv("C:/Users/satyam/Desktop/kaggle/new_titanic/train.csv")
df_test = pd.read_csv("C:/Users/satyam/Desktop/kaggle/new_titanic/test.csv")

my_imputer = Imputer()
survived_train = df_train.Survived
data = pd.concat([df_train.drop(['Survived'],axis=1),df_test])                                #concatenating both test and train datasets
data = pd.get_dummies(data,columns=['Sex'],drop_first=True)                                   #Handling categorical data
cols = ['Sex_male','Fare','Age','Pclass','SibSp']
numeric = data[cols]
filled_data = my_imputer.fit_transform(numeric)                                               #Filling missing values
data_train = filled_data[:891]
data_test= filled_data[891:]


model = XGBClassifier()                                                                       #XGBoost Model
model.fit(data_train,survived_train)
predictions = model.predict(data_test)                                                         #Making predictions

df_test['Survived']=predictions
df_test['Survived'] = df_test.Survived.apply(lambda x: int(x))
df_test[['PassengerId','Survived']].to_csv('C:/Users/satyam/Desktop/kaggle/new_titanic/XGBC_with_imputer.csv',index=False)

