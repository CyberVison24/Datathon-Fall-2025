# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 14:33:32 2025

@author: David
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#%%
data_file = 'battles_2020_12_09_use.csv'
battles_data = pd.read_csv(data_file)
print(battles_data.shape)
battles_data.head()

#%%
#Creates a new dataset with new columns

battles_data_1 = battles_data[['winner.troop.count', 'winner.structure.count', 'winner.spell.count', 'winner.elixir.average', 'loser.troop.count', 'loser.structure.count', 'loser.spell.count', 'loser.elixir.average']].copy()
battles_data_1.describe() 

#%%
#Winner-1 and loser-0
winner_df = battles_data_1[['winner.troop.count', 'winner.structure.count', 'winner.spell.count', 'winner.elixir.average']].copy()
winner_df['Winner'] = 1
#Rename Columns
winner_df.columns = ["troop_count","structure_count","spell_count", "elixir_average", "Winner"]
winner_df.head()

loser_df = battles_data_1[['loser.troop.count', 'loser.structure.count', 'loser.spell.count', 'loser.elixir.average']].copy()
#Creating Target Label 
loser_df['Winner'] = 0
#Rename Columns
loser_df.columns = ["troop_count","structure_count","spell_count", "elixir_average", "Winner"]
loser_df.head()

data_battles = pd.concat([loser_df,winner_df], ignore_index=True)
print(data_battles.isnull().sum())
data_battles.head()

#%%
#shuffling data to esnure the model learns a pattern
shuffled_data_battles = data_battles.sample(frac=1).reset_index(drop=True)
shuffled_data_battles.head()

#%%
#Establishing x and y 
X = shuffled_data_battles.drop('Winner',axis=1)
y = shuffled_data_battles['Winner']

#Use the train test split function
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42,train_size=0.7, shuffle=True)

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

model = XGBClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print(f"The accuracy of this model is {accuracy}")

battles_data_2 = battles_data[['winner.card1.id', 'winner.card2.id', 'winner.card3.id', 'winner.card4.id', 'winner.card5.id', 'winner.card6.id', 'winner.card7.id', 'winner.card8.id', 'loser.card1.id', 'loser.card2.id', 'loser.card3.id', 'loser.card4.id', 'loser.card5.id', 'loser.card6.id', 'loser.card7.id', 'loser.card8.id']].copy()
battles_data_2.describe() 

data2_columns = battles_data_2.columns

for col in data2_columns:
    battles_data_2[col] = battles_data_2[col].astype('category')
    print(col)

battles_data_2.describe()




