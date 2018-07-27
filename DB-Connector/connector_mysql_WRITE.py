#!/usr/bin/env python

import mysql.connector
from mysql.connector import errorcode

#connection
cnx = mysql.connector.connect(user='ObjOr86002', password='ObjOr86002',
                              host='objor86002.cjgzybuxaqpz.us-east-2.rds.amazonaws.com'
                              , database='TrialCreateDatabaseFromPython')

mycursor=cnx.cursor()

###################################
# 		create table and column
####################################

#mycursor.execute("USE TrialCreateDatabaseFromPython")
#mycursor.execute("""CREATE TABLE customer
#(
#		id int primary key,
#		name varchar (30),
#		email varchar (30),
#		city varchar (25),
#		age int,
#		gender char(1),
#		last_visit date
#		)

#	""")

###################################
# 			Insert Data
####################################

#mycursor.execute("""INSERT INTO customer VALUES
#	( 1,"Jack","jack@gmail.com","London",25,"M","2014-02-17")""")
#mycursor.execute("""INSERT INTO customer VALUES
#	( 2,"Jill","jill@gmail.com","New York",25,"F","2014-02-17")""")
#cnx.commit()

###################################
# 			Check Data
####################################

#mycursor.execute("SELECT * FROM customer")
#print(mycursor.fetchall())
#mylist=mycursor.fetchall()
#for x in mylist:
#	print(x)