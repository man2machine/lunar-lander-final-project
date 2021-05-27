# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 18:44:34 2020

@author: Shahir
"""

import os

import lander

def number_menu(option_list):
    print("-"*60)
    
    for n in range(len(option_list)):
        print(n, ": " , option_list[n])
    
    choice = input("Choose the number corresponding to your choice: ")
    for n in range(5):        
        try: 
            choice = int(choice)
            if choice < 0 or choice > len(option_list)-1:
                raise ValueError    
            print("-"*60 + "\n")
            return choice, option_list[choice]
        except ValueError: 
            choice = input("Invalid input, choose again: ")
    
    raise ValueError("Not recieving a valid input")

def get_rel_pkg_path(path):
    return os.path.abspath(os.path.join(os.path.dirname(lander.__file__), "..", path))
