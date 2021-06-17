import pandas as pd
from pandas import ExcelFile
from sklearn.model_selection import train_test_split as split
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor

df = pd.read_excel('.\\yourfilename.xlsx')
X=df.iloc[:,[1,2,3,4]]
y=df.iloc[:,5]

X_train, X_test, y_train, y_test = split(X, y, test_size=0.2, random_state=1)

print(len(X_train))

kVals = range(1, 20, 1)
accuracies = []

for k in kVals:
     
          model = AdaBoostRegressor(KNeighborsRegressor(n_neighbors=5),n_estimators=5,loss='exponential',learning_rate=1)
          model.fit(X_train, y_train)
          y_predicted=model.predict(X_test)
          score = model.score(X_test, y_test)
          print("k=%d, R^2 error=%.2f%%" % (k, score * 100))
          print("k=%d, mae = %.2f" % (k, mean_absolute_error(y_test,y_predicted)))
          print("k=%d, mse =%.2f%%" % (k, mean_squared_error(y_test,y_predicted) * 100))
          pred=np.array(y_predicted)
          org=np.array(y_test)
          dif= abs( pred- org )
          dif=1-dif/org
          div = pred/org
          ix=np.where(div>1)
          div[ix]=1/div[ix]
          div=1-div
          print("k=%d, mape =%.4f%%" % (k, dif.mean() * 100))
          print("k=%d, dist mmme accuracy =%.4f%%\n\n" % (k, div.mean() * 100))
          
                   
          accuracies.append(dif.mean())


i = np.argmax(accuracies)
print("k=%d achieved best mape of %.2f%% on validation data" % (kVals[i],
accuracies[i] * 100))


