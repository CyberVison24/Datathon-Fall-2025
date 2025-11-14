# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 08:06:56 2025

@author: bishi
"""

import pandas as pd
import matplotlib.pyplot as plt


file_path = "C:\\Users\\bishi\\OneDrive\\Documents\\Python Scripts\\g_test_results.csv"  
df = pd.read_csv(file_path)


col4 = df.iloc[:, 3]   

plt.figure(figsize=(8, 5))
plt.scatter(range(len(col4)), col4, color='blue')
plt.title("Scatter Plot of G test Statistics")
plt.xlabel("Row Index")
plt.ylabel("G test Statistics")
plt.grid(True)
plt.show()


col5 = df.iloc[:, 4]   

plt.figure(figsize=(10, 2))
plt.scatter(col5, [0]*len(col5), color='red')
plt.yticks([])          
plt.title("P values")
plt.xlabel("")
plt.grid(axis='x')
plt.show()