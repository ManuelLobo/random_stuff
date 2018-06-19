from sklearn.model_selection import train_test_split

x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_true,
                                            test_size=0.3, random_state=101)
