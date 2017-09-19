

```python
import numpy as np
import xgboost
from sklearn import cross_validation
from sklearn.metrics import accuracy_score

data = np.loadtxt('pima-indians-diabetes.csv',delimiter=',')
```


```python
X = data[:,0:8]
Y = data[:,8]
X.shape
```




    (768, 8)




```python
seed = 7
test_size = 0.33
X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(X, Y, test_size=test_size, random_state=seed)
```

### 训练模型


```python
model = xgboost.XGBClassifier()
model.fit(X_train,Y_train)
```




    XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
           gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3,
           min_child_weight=1, missing=None, n_estimators=100, nthread=-1,
           objective='binary:logistic', reg_alpha=0, reg_lambda=1,
           scale_pos_weight=1, seed=0, silent=True, subsample=1)



### 预测


```python
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
```

### 显示准确率


```python
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy: ",(accuracy * 100.0))
```

    Accuracy:  77.9527559055
    


```python
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,AdaBoostClassifier,AdaBoostRegressor
```


```python
model = RandomForestClassifier()
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy: ",(accuracy * 100.0))
```

    Accuracy:  74.4094488189
    


```python
model = RandomForestRegressor()
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy: ",(accuracy * 100.0))
```

    Accuracy:  74.0157480315
    


```python
model = AdaBoostClassifier()
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy: ",(accuracy * 100.0))
```

    Accuracy:  72.4409448819
    


```python
model = AdaBoostRegressor()
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy: ",(accuracy * 100.0))
```

    Accuracy:  75.5905511811
    


```python

```
