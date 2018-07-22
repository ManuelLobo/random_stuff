import psycopg2
from flask import Flask, jsonify
from helper_db import get_users, get_user_by_id

app = Flask(__name__)

conn = psycopg2.connect(host="localhost",database="post_db", user="postgres", password="post123")
cur = conn.cursor()


@app.route('/')
def hello_world():
    print("asjasajs")
    return 'Hello, World!'


@app.route('/users')
def users():
    users = get_users(cur)
    #print("hahaha")
    #return str(users)

    d = []
    for user in users:
        first_name = user[0]
        last_name = user[1]
        d.append({"first_name": first_name, "last_name": last_name})

    return jsonify(d)



@app.route('/users/<user_id>')
def show_user_profile(user_id):
    user = get_user_by_id(user_id, cur)
    first_name = user[0]
    last_name = user[1]
    d = {"first_name": first_name, "last_name": last_name}
    # show the user profile for that user

    return jsonify(d) #"{} {}".format(first_name, last_name)


#cur.close()
#conn.commit()
