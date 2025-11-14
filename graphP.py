# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 19:27:38 2025

@author: bishi
"""
import matplotlib.pyplot as plt

vector = [0, 0, 0, 0, 0, 0, 0, 0, 0.000001, 0, 0, 0, 0, 0.295702, 0, 0, 0.003937, 0.016226, 0, 0, 0.842705, 0, 0, 0.020921, 0, 0, 0.071787, 0, 0.015370, 0, 0, 0.606449, 0.000096, 0, 0, 0, 0, 0, 0, 0, 0.000005, 0, 0, 0, 0, 0, 0, 0.000013, 0, 0, 0.396434, 0, 0, 0, 0, 0.000021, 0, 0, 0, 0, 0.000021, 0, 0, 0, 0, 0.057171, 0, 0, 0, 0.024336, 0, 0, 0.195739, 0, 0, 0, 0, 0, 0.000248, 0, 0, 0, 0, 0.000167, 0, 0, 0.229180, 0.560166, 0, 0, 0, 0, 0, 0.010597, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.889096]
print(len(vector))


plt.figure(figsize=(4, 1))
plt.scatter(vector, [0]*len(vector), color='blue', s=100) 
plt.hlines(0, 0, max(vector), color='black', linewidth=1)
plt.yticks([]) 
plt.xlabel("Value")
plt.title("Number Line Plot")
plt.xlim(min(vector)-0.01, max(vector)+0.01)

plt.show()
