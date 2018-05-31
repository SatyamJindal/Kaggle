import pandas as pd                                                               
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.ensemble import RandomForestRegressor as rfr                             #Importing Libraries
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split as tts

path_tr = 'C:/Users/satyam/Desktop/kaggle/House_Prices/train.csv'                     #Training Data
data_tr = pd.read_csv(path_tr)

train_y = data_tr.SalePrice
cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']                        #Feature Selection
train_x = data_tr[cols]

model = rfr()                                                                         #Creating Model
model.fit(train_x,train_y)


path_te = 'C:/Users/satyam/Desktop/kaggle/House_Prices/test.csv'                      #Test Data
data_te = pd.read_csv(path_te)

test_x = data_te[cols]

predictions = model.predict(test_x)                                                   #Making Prections

my_submission = pd.DataFrame({'Id': data_te.Id, 'SalePrice': predictions})

my_submission.to_csv('Random_forest.csv',index=False)

