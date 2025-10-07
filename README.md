# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Required Libraries
2. Data Reading and Preprocessing
3.Adding Bias and Initializing Parameters
4.Gradient Descent Parameter Update
5.Prediction on New Data
 
## Program:
```

Program to implement the linear regression using gradient descent.
Developed by: GAYATHRI.C
RegisterNumber: 25009125

import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler 
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000): 
  X = np.c_[np.ones(len(X1)),X1] 
  theta = np.zeros(X.shape[1]).reshape(-1,1) 
  for _ in range(num_iters): 
    predictions = (X).dot(theta).reshape(-1,1) 
    errors=(predictions - y ).reshape(-1,1) 
    theta -= learning_rate*(1/len(X1))*X.T.dot(errors) 
  return theta 
data=pd.read_csv("50_Startups.csv") 
print(data.head()) 
print("\n") 
X=(data.iloc[1:,:-2].values) 
X1=X.astype(float) 
scaler=StandardScaler() 
y=(data.iloc[1:,-1].values).reshape(-1,1) 
X1_Scaled=scaler.fit_transform(X1) 
Y1_Scaled=scaler.fit_transform(y) 
print(X) 
print("\n") 
print(X1_Scaled) 
print("\n") 
theta= linear_regression(X1_Scaled,Y1_Scaled) 
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1, 1) 
new_Scaled=scaler.fit_transform(new_data) 
prediction=np.dot(np.append(1,new_Scaled),theta) 
prediction=prediction.reshape(-1,1) 
pre=scaler.inverse_transform(prediction) 
print(prediction) 
print(f"Predicted value: {pre}")
```

## Output:
## DATA INFO:
<img width="643" height="132" alt="Screenshot 2025-10-07 071322" src="https://github.com/user-attachments/assets/a57f73c3-2acb-4b4f-b75a-7d41b3376d1b" />

 ## VALUE OF X:
<img width="278" height="845" alt="Screenshot 2025-10-07 071316" src="https://github.com/user-attachments/assets/d344b7ee-eabe-42cb-8909-440251c0635f" />

## VALUE OF X_SCALED:
<img width="418" height="842" alt="Screenshot 2025-10-07 071350" src="https://github.com/user-attachments/assets/d23042bd-8708-40e3-b723-0b84d51b5858" />

## PRDECITED VALUE:
<img width="344" height="52" alt="Screenshot 2025-10-07 071401" src="https://github.com/user-attachments/assets/bea188af-039d-4daf-b91d-9e18d7391bad" />





## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
