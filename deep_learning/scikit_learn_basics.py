import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# Scaing data
data = np.random.randint(0, 100, (10, 2))
scalar_model = MinMaxScaler()
scalar_model.fit(data)
scaled = scalar_model.transform(data)
print(scaled)

scaled = scalar_model.fit_transform(data) #fit and transform in 1 step.
print(scaled)


# Train/Test Split
data = np.random.randint(0, 101, (50, 4))
dataframe = pd.DataFrame(data=data)
print(dataframe)

dataframe = pd.DataFrame(data=data, columns=["feat_1", "feat_2", "feat_3", "label"])
print(dataframe)

X = dataframe[["feat_1", "feat_2", "feat_3"]]
y = dataframe["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=101)


# Train/test set sizes
print(f"Training Input Set Shape: {X_train.shape}")
print(f"Training Label Set Shape: {y_train.shape}")
print(f"Testing Input Set Shape: {X_test.shape}")
print(f"Testing Label Set Shape: {y_test.shape}")
