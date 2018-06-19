import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

housing = pd.read_csv("housing.csv")

x_data = housing.drop("medianHouseValue", axis=1)
y_val = housing["medianHouseValue"]


X_train, X_test, y_train, y_test = train_test_split(x_data, y_val,
                                                    test_size=0.3,
                                                    random_state=101)


# Scale the feature data - Fit the scalar to the training data. Then use it to
# transform X_test and X_train.
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = pd.DataFrame(data=scaler.transform(X_train), columns=X_train.columns,
                       index=X_train.index)
X_test = pd.DataFrame(data=scaler.transform(X_test), columns=X_test.columns,
                      index=X_test.index)


# Create feature columns
age = tf.feature_column.numeric_column('housingMedianAge')
rooms = tf.feature_column.numeric_column('totalRooms')
bedrooms = tf.feature_column.numeric_column('totalBedrooms')
pop = tf.feature_column.numeric_column('population')
households = tf.feature_column.numeric_column('households')
income = tf.feature_column.numeric_column('medianIncome')

feature_columns = [age, rooms, bedrooms, pop, households, income]

# Create input function for the estimator object
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train,
                                              batch_size=10, num_epochs=1000,
                                              shuffle=True)

# Create the estimator model using DNNRegressor
model = tf.estimator.DNNRegressor(hidden_units=[6, 6, 6],
                                  feature_columns=feature_columns)

# Train the model for 20000 steps
model.train(input_fn=input_func, steps=20000)

# Create the prediction input function and predict on the test data
predict_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,
                                                         batch_size=10,
                                                         num_epochs=1,
                                                         shuffle=False)

prediction_generator = model.predict(predict_input_func)
predictions = list(prediction_generator)


# Calculate the RMSE (Root Mean Square Error). Should be around 100,000
final_predictions = []
for pred in prediction_generator:
    final_predictions.append(pred["predictions"])

RMSE = mean_squared_error(y_test, final_predictions)**0.5
print("Root Mean Squared Error:", RMSE)
