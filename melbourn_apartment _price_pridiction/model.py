import pandas as pan
from sklearn.tree import DecisionTreeRegressor 


filePath="c:\\Users\\navcer\\Downloads\\Compressed\\melb_data.csv\\melb_data.csv"
data=pan.read_csv(filePath)
data=data.dropna(axis=0)                                            #drop missing rows 
Y=data.Price                                                        #prediction target
melbournePredictors=['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X=data[melbournePredictors]                                         #Predictors
melbournModel=DecisionTreeRegressor()
melbournModel.fit(X,Y)                                              #data fiting of price column in Predictors colm

print("Making predictions for the following 5 houses:")

print(X.head())      
                                                  

print("The predictions are")

predictes=melbournModel.predict(X.head())                               # predicting the X.head() part      

for price in predictes:       
    print(price) 

