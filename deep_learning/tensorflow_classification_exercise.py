import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split



census = pd.read_csv("census.csv")

# String labels need to be converted to 0 and 1. pandas has the funcion apply()
# which converts the strings.

# the label is "income_bracket"
print(census["income_bracket"].unique()) # all unique labels

def label_fix(label):
    if label == ' <=50K':
        return 0
    else:
        return 1

# apply created function
census["income_bracket"] = census["income_bracket"].apply(label_fix)
print(census["income_bracket"].unique())

# train_test_split
x_data = census.drop('income_bracket',axis=1)
y_labels = census['income_bracket']
X_train, X_test, y_train, y_test = train_test_split(x_data,y_labels,test_size=0.3,random_state=101)


# Create Feature Columens for tf.estimator

# vocabulary list - list of possibilities
gender = tf.feature_column.categorical_column_with_vocabulary_list("gender", ["Female", "Male"])

# hash buckets, put a size according to the number of options.
occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size=1000)
marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital_status", hash_bucket_size=1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship", hash_bucket_size=1000)
education = tf.feature_column.categorical_column_with_hash_bucket("education", hash_bucket_size=1000)
workclass = tf.feature_column.categorical_column_with_hash_bucket("workclass", hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket("native_country", hash_bucket_size=1000)

# Numerical columns
age = tf.feature_column.numeric_column("age")
education_num = tf.feature_column.numeric_column("education_num")
capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")

# create list with all variables
feat_cols = [gender,occupation,marital_status,relationship,education,workclass,native_country,
            age,education_num,capital_gain,capital_loss,hours_per_week]


# Create input function

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train,
                                                 batch_size=100, num_epochs=None,
                                                 shuffle=True)


# Create the model (linear classifier or DNN classifier)
# To use DNN classifier, the variables would need to be converted to
# embedded colimns out of the categorical feature that use strings.

# Linear classifier
model = tf.estimator.LinearClassifier(feature_columns=feat_cols)
model.train(input_fn=input_func,steps=5000)

# Evaluation - Prediction function
pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)


predictions = list(model.predict(input_fn=pred_fn))


predictions[0]

final_preds = []
for pred in predictions:
    final_preds.append(pred['class_ids'][0])
final_preds[:10]

# Report on the performance of the model
from sklearn.metrics import classification_report
print(classification_report(y_test,final_preds))





#
