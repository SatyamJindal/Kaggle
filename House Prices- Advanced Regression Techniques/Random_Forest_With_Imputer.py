#Score - 0.15891


pandas as pd
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.ensemble import RandomForestRegressor as rfr                                         #Importing Libraries
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import Imputer

path_tr = 'C:/Users/satyam/Desktop/kaggle/House_Prices/train.csv'
train = pd.read_csv(path_tr)
path_te = 'C:/Users/satyam/Desktop/kaggle/House_Prices/test.csv'
test = pd.read_csv(path_te)

target = train.SalePrice
data = pd.concat([train.drop(['SalePrice'],axis=1),test])
numeric = data.select_dtypes(exclude=['object'])                                                  #Using only numerical predictors
my_imputer = Imputer()
filled_data = my_imputer.fit_transform(numeric)                                                   #Using Imputer for handling Null values
train_f = filled_data[:1460]
test_f = filled_data[1460:]

model = rfr()                                                                                     #Using Random Forest Model
model.fit(train_f,target)

predictions = model.predict(test_f)                                                               #Making predictions
my_submission = pd.DataFrame({'Id':test.Id,'SalePrice':predictions})
my_submission.to_csv('Random_forest_wth_Imputer.csv',index=False)

