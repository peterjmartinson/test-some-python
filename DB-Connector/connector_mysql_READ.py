# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 20:45:09 2018

@author: jiahaoyuan
"""

#!/usr/bin/env python

import mysql.connector
from mysql.connector import errorcode

#connection
cnx = mysql.connector.connect(user='ObjOr86002', password='ObjOr86002',
                              host='objor86002.cjgzybuxaqpz.us-east-2.rds.amazonaws.com'
                              , database='SampleDB')

mycursor=cnx.cursor()

# GET DIAGNOSIS
# PARSE IT TO DIAGNOSIS, WHICH IS A LIST
query = ("SELECT diagnosis FROM UW_Data")
mycursor.execute(query)

diagnosis=[]
for d in mycursor:
    for i in d:
        diagnosis.append(i)

print(diagnosis)
#print(diagnosis[0])

# GET EVERYTHING BUT DIAGNOSIS
# DATA STRUCTURE: LIST IN LIST
query = ("SELECT * FROM UW_Data")
mycursor.execute(query)

data=[]
for a in mycursor:
    patient=[]
    for i in a:
        patient.append(i)
    # DELETE THE DIAGNOSIS
    del patient[1]
    data.append(patient)

print(data)
print(data[0])
