# -*- coding: utf-8 -*-
"""
Created on Sun May 26 00:53:39 2019

@author: peng.zhou
"""
import requests
a=input("please input a count:\n")
a=int(a)
print("the {} now is inputted".format(a))
url_1="http://192.168.43.20:8080/1"
url_2="http://192.168.43.20:8080/2"
if  a==1:
    print("z1")
    r = requests.post(url_1)
    print("z2")
    result = r.text
    print (result)
elif a==2:
    r = requests.post(url_2)
    result = r.text
    print (result)

    
#10.135.11.39
