import csv


def prepare_data(input_file):
    users = []
    with open(input_file) as infile:
        reader = csv.reader(infile)
        for line in reader:
            first_name = line[1]
            last_name = line[2]
            users.append((first_name, last_name))
    return users


def insert_user(first_name, last_name, cursor):
    query = """INSERT INTO users (first_name, last_name)
               VALUES ('{}', '{}');""".format(first_name, last_name)

    print("inserting user {} {}".format(first_name, last_name))
    cursor.execute(query)


def get_users(cursor):
    query = "SELECT first_name, last_name FROM users;"
    cursor.execute(query)
    users = cursor.fetchall()
    return users


def get_user_by_id(user_id, cursor):
    query = "SELECT first_name, last_name FROM users WHERE user_id = {};".format(user_id)
    cursor.execute(query)
    user = cursor.fetchone()
    return user


def create_default_table(cursor):
    query = """CREATE TABLE users (
               user_id serial PRIMARY KEY,
               first_name varchar (50) NOT NULL,
               last_name varchar (50) NOT NULL);"""

    cursor.execute(query)
