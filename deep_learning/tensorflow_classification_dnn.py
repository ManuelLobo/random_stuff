import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# We're trying to predict the class (1 or 0)
diabetes = pd.read_csv("diabetes.csv")
#print(diabetes.head())

# first we need to clean the data (normalize)
diabetes_columns = diabetes.columns
print(diabetes_columns)
columns_to_normalize = ['Number_pregnant', 'Glucose_concentration',
                        'Blood_pressure', 'Triceps', 'Insulin', 'BMI',
                        'Pedigree']

# normalize columns with pandas
diabetes[columns_to_normalize] = diabetes[columns_to_normalize].apply(
    lambda x: (x - x.min() / (x.max() - x.min() )))

print(diabetes.head())

# we need to create feature columns for each of the columns
# For continuous values
number_pregnant = tf.feature_column.numeric_column("Number_pregnant")
glucose_concentration = tf.feature_column.numeric_column("Glucose_concentration")
blood_pressure = tf.feature_column.numeric_column("Blood_pressure")
triceps = tf.feature_column.numeric_column("Triceps")
insulin = tf.feature_column.numeric_column("Insulin")
bmi = tf.feature_column.numeric_column("BMI")
pedigree = tf.feature_column.numeric_column("Pedigree")
age = tf.feature_column.numeric_column("Age")

# For categorical values: We can do it using a vocabulary list or a hash bucket

# Vocabulary list
assigned_group = tf.feature_column.categorical_column_with_vocabulary_list(
    "Group", ["A", "B", "C", "D"])

# Hash Bucket
#assigned_group = tf.feature_column.categorical_column_with_hash_bucket(
#    "Group", hash_bucket_size=10)

diabetes["Age"].hist(bins=20)
#plt.show()

age_bucket = tf.feature_column.bucketized_column(age,
                        boundaries=[20,30,40,50,60,70,80])

feat_cols = [number_pregnant, glucose_concentration, blood_pressure, triceps,
             insulin, bmi, pedigree, assigned_group, age_bucket]

# Train Test Split

x_data = diabetes.drop("Class", axis=1)
labels = diabetes["Class"]


X_train, X_test, y_train, y_test = train_test_split(x_data, labels,
                                            test_size=0.3, random_state=101)


# For categorical columns it's necessary to embed
embedded_group_column = tf.feature_column.embedding_column(assigned_group,
                                                           dimension=4)
feat_cols = [number_pregnant, glucose_concentration, blood_pressure, triceps,
             insulin, bmi, pedigree, embedded_group_column, age_bucket]

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train,
    batch_size=10, num_epochs=1000, shuffle=True)

dnn_model = tf.estimator.DNNClassifier(hidden_units=[10,10,10],
    feature_columns=feat_cols, n_classes=2)

dnn_model.train(input_fn=input_func, steps=1000)

eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test,
    batch_size=10, num_epochs=1, shuffle=False)

results = dnn_model.evaluate(eval_input_func)
print(results)

    #
