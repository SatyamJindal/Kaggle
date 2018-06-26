'''

Error - 0.14566

'''



import pandas as pd
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.ensemble import RandomForestRegressor as rfr
from xgboost import XGBRegressor as xgb                                                         #Importing Modules
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import Imputer


path_tr = 'C:/Users/satyam/Desktop/kaggle/House_Prices/train.csv'
train = pd.read_csv(path_tr)                                                                    #Data
path_te = 'C:/Users/satyam/Desktop/kaggle/House_Prices/test.csv'
test = pd.read_csv(path_te)

my_imputer = Imputer()
target = train.SalePrice                                                            
data = pd.concat([train.drop(['SalePrice'],axis=1),test])                                       
numeric = data.select_dtypes(exclude=['object'])

filled_data = my_imputer.fit_transform(numeric)                                                 #Using Imputer to fill up missing values
train_f = filled_data[:1460]
test_f = filled_data[1460:]

model = xgb(n_estimators=1000,learning_rate=0.05)                                               #Using XGBoost Model along with estimator and learning rate
model.fit(train_f,target,early_stopping_rounds=5,eval_set=[(train_f,target)],verbose=False)

predictions = model.predict(test_f)                                                             #Making predictions
my_submission = pd.DataFrame({'Id':test.Id,'SalePrice':predictions})
my_submission.to_csv('XGBoost+Imputer+est+LR+ESR.csv',index=False)
