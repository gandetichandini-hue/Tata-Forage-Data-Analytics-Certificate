import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df=pd.read_csv('churn_data.csv')
X=df.drop('Churn',axis=1)
y=df['Churn']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LogisticRegression(max_iter=200)
model.fit(X_train,y_train)
pred=model.predict(X_test)
print("Accuracy:",accuracy_score(y_test,pred))
