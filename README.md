# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import Libraries: pandas,numpy,matplotlib,sklearn.
2. Load Dataset: Read CSV file containing study hours and marks.
3. Check Data: Preview data and check for missing values.
4. Define Variables: Set x=Hours, y=Scores.
5. Split Data: Train-test spilt(80-20).
6. Train Model: Fit Linear Regression on training data.
7. Predict: use model to predict scores on test data.
8. Evaluate: Calculate Mean Absolute Error(MAE) and R^2 score.
9. Visualize:Plot actual data and regression line. 

## Program:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
data=pd.read_csv("/content/student_scores.csv")
print("Dataset Preview:\n",data.head())
print("\nMissing Values:\n",data.isnull().sum())
x=data[['Hours']]
y=data[['Scores']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("\nIntercept:",model.intercept_)
print("Slope:",model.coef_[0])
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("\nMean Absolute Error:",mae)
print("R^2 Score:",r2)
plt.scatter(x,y,color='blue',label="Actual Data")
plt.plot(x,model.predict(x),color='red',label='Regression Line')
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("simple Linear Regression - Marks Prediction")
plt.legend()
plt.show() 
```

## Output:
![Screenshot 2025-04-19 220631](https://github.com/user-attachments/assets/5d2f5acc-0cb9-4d22-a510-025dcf9b9a57)
![Screenshot 2025-04-19 220652](https://github.com/user-attachments/assets/b89a0ece-579e-4935-870f-0780863e29b4)
![Screenshot 2025-04-19 220708](https://github.com/user-attachments/assets/b00c98b1-6afa-41e7-9268-c4557ce29636)
![Screenshot 2025-04-19 220725](https://github.com/user-attachments/assets/08870e3f-4f98-4d6d-9d85-cf77f1d6ff7c)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
