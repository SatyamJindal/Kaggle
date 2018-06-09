# Score - 0.14231


import pandas as pd
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.metrics import mean_absolute_error as mae                                          #Importing Libraries
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor


path_tr = 'C:/Users/satyam/Desktop/kaggle/House_Prices/train.csv'
train = pd.read_csv(path_tr)
path_te = 'C:/Users/satyam/Desktop/kaggle/House_Prices/test.csv'
test = pd.read_csv(path_te)

my_imputer = Imputer()                                                                        #Using Imputer for handling Null values
target = train.SalePrice
data = pd.concat([train.drop(['SalePrice'],axis=1),test])
numeric = data.select_dtypes(exclude=['object'])                                              #Using only numerical predictors
filled_data = my_imputer.fit_transform(numeric)
train_f = filled_data[:1460]
test_f = filled_data[1460:]

model = XGBRegressor()                                                                        #Using XGBoost Model
model.fit(train_f,target,verbose=False)

predictions = model.predict(test_f)
my_submission = pd.DataFrame({'Id':test.Id,'SalePrice':predictions})                          #Making predictions
my_submission.to_csv('XGBoost_wth_Imputer.csv',index=False)
