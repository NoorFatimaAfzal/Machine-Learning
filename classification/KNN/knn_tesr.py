import numpy as np
import pandas as pd
from KNearestNeighnour import KNearestNeighbours

data = pd.read_csv(r'C:\Users\InfoBay\OneDrive\Desktop\Machine_Learning\classification\KNN\Social_Network_Ads.csv')

X = data.iloc[: , 2:4].values
y = data.iloc[: , -1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

knn=KNearestNeighbours(k=5)

knn.fit(X_train,y_train)

def predict_new():
    age=int(input("Enter the age"))
    salary=int(input("Enter the salary"))
    X_new=np.array([[age],[salary]]).reshape(1,2)

    X_new=scalar.transform(X_new)

    result=knn.predict(X_new)

    if result==0:
        print("Will not purchase")
    else:
        print("Will purchase")

predict_new()

