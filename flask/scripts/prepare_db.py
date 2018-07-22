import psycopg2
import csv





if __name__ == '__main__':
    conn = psycopg2.connect(host="localhost",database="post_db", user="postgres", password="post123")
    cur = conn.cursor()
    # cur.execute('SELECT version()')
    #db_version = cur.fetchone()
    #print(db_version)

    # users = prepare_data("data.tsv")
    # for user in users:
    #     #print(user)
    #     insert_user(user[0], user[1], cur)

    #create_default_table(cur)

    #users = get_users(cur)
    user = get_user_by_id("3", cur)
    print(user)


    cur.close()
    conn.commit()




# def connect():
#     """ Connect to the PostgreSQL database server """
#     conn = None
#     try:
#         # read connection parameters
#         params = config()
#
#         # connect to the PostgreSQL server
#         print('Connecting to the PostgreSQL database...')
#         conn = psycopg2.connect(**params)
#
#         # create a cursor
#         cur = conn.cursor()
#
#  # execute a statement
#         print('PostgreSQL database version:')
#         cur.execute('SELECT version()')
#
#         # display the PostgreSQL database server version
#         db_version = cur.fetchone()
#         print(db_version)
#
#      # close the communication with the PostgreSQL
#         cur.close()
#     except (Exception, psycopg2.DatabaseError) as error:
#         print(error)
#     finally:
#         if conn is not None:
#             conn.close()
#             print('Database connection closed.')



#
# INSERT INTO playground (type, color, location, install_date) VALUES ('slide', 'blue', 'south', '2014-04-28');
# INSERT INTO playground (type, color, location, install_date) VALUES ('swing', 'yellow', 'northwest', '2010-08-16');
#
#
# SELECT * FROM playground;
# DELETE FROM playground WHERE type = 'slide';
# SELECT * FROM playground;
#
#
# ALTER TABLE playground ADD last_maint date;
