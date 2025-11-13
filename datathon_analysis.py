# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 13:03:28 2025

@author: David
"""

import pandas as pd

#%% 1) Load the data
file_path = "C:/Users\David/Documents/Fall 2025/datathon/battles_2020_12_09_use.csv"
battles_df = pd.read_csv(file_path)

# Check the structure
print(battles_df.shape) # (47881, 61)
print(battles_df.head())

'''
                  battleTime  ...  loser.elixir.average
0  2020-12-09 15:40:02+00:00  ...                 3.625
1  2020-12-09 15:09:05+00:00  ...                 3.875
2  2020-12-09 04:20:22+00:00  ...                 3.500
3  2020-12-09 01:28:08+00:00  ...                 4.250
4  2020-12-09 22:57:54+00:00  ...                 4.375

[5 rows x 61 columns]
'''

# Identify the card columns (should be 8 cards per player)
card_cols = [c for c in battles_df.columns if "card" in c and ".id" in c]
print("Card columns:", card_cols)

'''
Card columns: 
    ['winner.card1.id', 'winner.card2.id', 'winner.card3.id', 
     'winner.card4.id', 'winner.card5.id', 'winner.card6.id',
     'winner.card7.id', 'winner.card8.id',
     
     'loser.card1.id', 'loser.card2.id', 'loser.card3.id', 'loser.card4.id',
     'loser.card5.id', 'loser.card6.id', 'loser.card7.id', 'loser.card8.id']
'''

#%% 2) Create the deck categories (archetypes)
import numpy as np

def detect_archetype(cards):
   # Combine the 8 cards into a single string for quick checks
   deck = " ".join(cards)
   
   # Define rules with capitalized card names
   if any(x in deck for x in ["X-Bow", "Mortar"]):
       return "Siege"
   elif any(x in deck for x in ["Golem", "Giant", "Lava Hound", "P.E.K.K.A", "Electro Giant"]):
       return "Beatdown"
   elif any(x in deck for x in ["Hog Rider"]) and not any(x in deck for x in ["Golem", "Giant"]):
       return "Cycle"
   elif any(x in deck for x in ["Goblin Barrel", "Princess", "Goblin Gang", "Skeleton Army"]):
       return "Bait"
   elif any(x in deck for x in ["Bandit", "Ram Rider", "Battle Ram"]):
       return "Bridgespam"
   else:
       return "Control"  # fallback type

# Apply to each player's deck (winner and loser combined dataset)
player_data = []

for i, row in battles_df.iterrows():
    for side in ["winner", "loser"]:
        cards = [str(row.get(f"{side}.card{i}.id", "")) for i in range(1, 9)]
        player_data.append({
            "avg_elixir": row.get(f"{side}.elixir.average", np.nan),
            "Winner": 1 if side == "winner" else 0,
            "archetype": detect_archetype(cards)
        })

players = pd.DataFrame(player_data)
print(players.head())

'''
   avg_elixir  Winner   archetype
0       3.875       1    Beatdown
1       3.625       0     Control
2       4.125       1       Cycle
3       3.875       0        Bait
4       4.250       1  Bridgespam
'''

#%% 3) Create distributions & descriptive summaries

import matplotlib.pyplot as plt
import seaborn as sns

# Archetype distribution
plt.figure(figsize=(8,5))
sns.countplot(x="archetype", data=players, order=players["archetype"].value_counts().index)
plt.title("Deck Archetype Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Average elixir distribution by archetype
plt.figure(figsize=(8,5))
sns.boxplot(x="archetype", y="avg_elixir", data=players, order=players["archetype"].value_counts().index)
plt.title("Average Elixir by Archetype")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Summary table
summary = players.groupby("archetype")["avg_elixir"].describe()
print(summary)

'''
              count      mean       std       min    25%    50%    75%    max
archetype                                                                    
Bait        15131.0  3.635835  0.454759  2.125000  3.250  3.625  4.000  5.250
Beatdown    53017.0  3.914373  0.507589  1.500000  3.625  3.875  4.250  7.500
Bridgespam   2608.0  3.858375  0.445980  2.375000  3.500  3.875  4.125  5.500
Control     10518.0  3.738358  0.560656  1.500000  3.375  3.750  4.125  6.125
Cycle        9183.0  3.707132  0.459664  1.839286  3.375  3.750  4.000  6.000
Siege        5305.0  3.338454  0.454496  2.500000  3.000  3.125  3.500  6.625
'''

#%% 3b) Create distributions & descriptive summaries (with color)
# Define a consistent color mapping for each archetype
archetype_colors = {
    "Siege": "#FF6F61",       # reddish
    "Beatdown": "#6B5B95",    # purple
    "Cycle": "#88B04B",       # green
    "Bait": "#F7CAC9",        # pink
    "Bridgespam": "#92A8D1",  # light blue
    "Control": "#955251"      # brown
}

# Make a list of colors in the order of value_counts
archetype_order = players["archetype"].value_counts().index
colors_ordered = [archetype_colors[a] for a in archetype_order]

# 1) Deck Archetype Distribution
plt.figure(figsize=(8,5))
sns.countplot(
    x="archetype",
    data=players,
    order=archetype_order,
    palette=colors_ordered
)
plt.title("Deck Archetype Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2) Average Elixir by Archetype
plt.figure(figsize=(8,5))
sns.boxplot(
    x="archetype",
    y="avg_elixir",
    data=players,
    order=archetype_order,
    palette=colors_ordered
)
plt.title("Average Elixir by Archetype")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% 4) Build model (Linear and XGboost)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# One-hot encode archetypes
encoder = OneHotEncoder(drop='first')
arch_ohe = encoder.fit_transform(players[["archetype"]]).toarray()
arch_cols = encoder.get_feature_names_out(["archetype"])

X = pd.DataFrame(arch_ohe, columns=arch_cols)
X["avg_elixir"] = players["avg_elixir"]
y = players["Winner"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

'''
Logistic Regression Accuracy: 0.5130356086184692
              precision    recall  f1-score   support

           0       0.51      0.54      0.53     14375
           1       0.51      0.48      0.50     14354

    accuracy                           0.51     28729
   macro avg       0.51      0.51      0.51     28729
weighted avg       0.51      0.51      0.51     28729
'''

# XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

'''
XGBoost Accuracy: 0.5335723484980334
              precision    recall  f1-score   support

           0       0.55      0.40      0.46     14375
           1       0.53      0.66      0.59     14354

    accuracy                           0.53     28729
   macro avg       0.54      0.53      0.53     28729
weighted avg       0.54      0.53      0.53     28729
'''

# Feature importance
importances = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importances (XGBoost):\n", importances)
importances.plot(kind="barh", figsize=(8,5))
plt.title("Feature Importance (XGBoost)")
plt.tight_layout()
plt.show()

'''
Feature Importances (XGBoost):
 archetype_Bridgespam    0.248427
archetype_Siege         0.192345
archetype_Control       0.152396
archetype_Cycle         0.140016
avg_elixir              0.139227
archetype_Beatdown      0.127590
dtype: float32
'''

#%% 5) top 10 most cards

# Step 1: Collect all card ID columns (winner + loser) and rename variable
top10card_cols = [c for c in battles_df.columns if "card" in c and ".id" in c]

# Step 2: Flatten all card IDs into a single series
all_cards = battles_df[top10card_cols].astype(str).melt(value_name="card_id")["card_id"]

# Step 3: Count occurrences of each card ID
card_counts = all_cards.value_counts().reset_index()
card_counts.columns = ["card_id", "count"]

# Step 4: Take the top 10 most used cards
top10_cards = card_counts.head(10)
print("Top 10 Most Used Cards:")
print(top10_cards)

'''
Top 10 Most Used Cards:
         card_id  count
0            Zap  28244
1        The Log  28056
2       Fireball  25723
3       Valkyrie  22884
4         Wizard  20458
5         Arrows  19543
6         Knight  19145
7      Hog Rider  18970
8  Skeleton Army  18209
9    Mega Knight  15918
'''

# Step 5: Plot the counts
plt.figure(figsize=(8,5))
sns.barplot(x="count", y="card_id", data=top10_cards, palette="crest")
plt.title("Top 10 Most Used Cards")
plt.xlabel("Usage Count")
plt.ylabel("Card ID")
plt.tight_layout()
plt.show()














