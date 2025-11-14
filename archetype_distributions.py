# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 13:03:28 2025

@author: David
"""


import pandas as pd

#%% 1) Load the data
file_path = "C:/Users\David/Documents/Fall 2025/datathon/battles_2020_12_09.csv"
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

#%% 3) Create archetype distributions & descriptive summaries

import matplotlib.pyplot as plt
import seaborn as sns

# Archetype distribution
plt.figure(figsize=(8,5))
sns.countplot(x="archetype", data=players, order=players["archetype"].value_counts().index)
plt.title("Deck Archetype Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Count total number of decks per archetype
archetype_counts = players["archetype"].value_counts().reset_index()
archetype_counts.columns = ["archetype", "total_decks"]

# add percentage of total decks
total = archetype_counts["total_decks"].sum()
archetype_counts["percentage"] = (archetype_counts["total_decks"] / total * 100).round(2)

# Sort by total decks (or keep original order)
archetype_counts = archetype_counts.sort_values(by="total_decks", ascending=False).reset_index(drop=True)

print(archetype_counts)

'''
    archetype  total_decks  percentage
0    Beatdown        53017       55.36
1        Bait        15131       15.80
2     Control        10518       10.98
3       Cycle         9183        9.59
4       Siege         5305        5.54
5  Bridgespam         2608        2.72
'''

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

#%% 3b) Create archetype distributions (with color)

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

import matplotlib.ticker as mtick
from matplotlib.ticker import ScalarFormatter

plt.figure(figsize=(8,5))
ax = sns.countplot(
    x="archetype",
    data=players,
    order=archetype_order,
    palette=colors_ordered
)

# Set title and labels
plt.title("Deck Archetype Distribution")
plt.xlabel("Archetype")
plt.ylabel("Number of Decks")
plt.xticks(rotation=45)

# --- Change y-axis scale ---
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))  # scientific notation
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))     # force scientific notation

# Optional: set custom ticks
# ax.set_yticks([0, 10000, 20000, 30000, 40000, 50000])

plt.tight_layout()
plt.show()

#%% 3c) Create distributions by elixir (with color)

# Count number of decks per archetype by winner/loser
archetype_counts = players.groupby(["archetype", "Winner"]).size().reset_index(name="count")
print(archetype_counts)

plt.figure(figsize=(10,6))
ax = sns.barplot(
    x="archetype",
    y="count",
    hue="Winner",          # 0 = loser, 1 = winner
    data=archetype_counts,
    palette={1: "green", 0: "red"}
)
plt.title("Archetype Decks (Winners vs Losers) ")
plt.xlabel("Archetype")
plt.ylabel("Number of Decks")
plt.xticks(rotation=45)

# Update legend labels
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ["Loser", "Winner"], title="Deck Outcome")  # custom legend

# Optional: scientific notation for y-axis
from matplotlib.ticker import ScalarFormatter
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

plt.tight_layout()
plt.show()

#%% 3D) archetype by exlier

import matplotlib.pyplot as plt

# Compute mean per archetype (ignore std if you don't want error bars)
elixir_summary = players.groupby("archetype")["avg_elixir"].mean().reindex(archetype_order)

plt.figure(figsize=(10,6))
plt.bar(
    elixir_summary.index,
    elixir_summary.values,
    color=[archetype_colors[a] for a in elixir_summary.index]  # use your color mapping
)
plt.title("Archetype Average Elixir")
plt.xlabel("Archetype")
plt.ylabel("Average Elixir")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(elixir_summary)
'''
archetype
Beatdown      3.914373
Bait          3.635835
Control       3.738358
Cycle         3.707132
Siege         3.338454
Bridgespam    3.858375
Name: avg_elixir, dtype: float64
'''

#%% 4) top 15 most cards

# Step 1: Collect all card ID columns (winner + loser) and rename variable
top10card_cols = [c for c in battles_df.columns if "card" in c and ".id" in c]

# Step 2: Flatten all card IDs into a single series
all_cards = battles_df[top10card_cols].astype(str).melt(value_name="card_id")["card_id"]

# Step 3: Count occurrences of each card ID
card_counts = all_cards.value_counts().reset_index()
card_counts.columns = ["card_id", "count"]

# Step 4: Take the top 10 most used cards
top10_cards = card_counts.head(15)
print("Top 15 Most Used Cards:")
print(top10_cards)

'''
Top 15 Most Used Cards:
           card_id  count
0              Zap  28244
1          The Log  28056
2         Fireball  25723
3         Valkyrie  22884
4           Wizard  20458
5           Arrows  19543
6           Knight  19145
7        Hog Rider  18970
8    Skeleton Army  18209
9      Mega Knight  15918
10     Baby Dragon  15601
11  Electro Wizard  15528
12  Mini P.E.K.K.A  14216
13            Bats  14166
14         Balloon  13905
'''

# Step 5: Plot the counts
plt.figure(figsize=(8,5))
sns.barplot(x="count", y="card_id", data=top10_cards, palette="crest")
plt.title("Top 15 Most Used Cards")
plt.xlabel("Usage Count")
plt.ylabel("Card ID")
plt.tight_layout()
plt.show()

#%% 5b) side by side comparison

# Identify card columns
winner_cols = [c for c in battles_df.columns if c.startswith("winner.") and ".card" in c and ".id" in c]
loser_cols  = [c for c in battles_df.columns if c.startswith("loser.") and ".card" in c and ".id" in c]

# Flatten to long form
winner_cards = battles_df[winner_cols].melt(value_name="card_id")["card_id"].astype(str)
loser_cards  = battles_df[loser_cols].melt(value_name="card_id")["card_id"].astype(str)

# Frequencies
winner_counts = winner_cards.value_counts().head(15)
loser_counts  = loser_cards.value_counts().head(15)

print(winner_counts, loser_counts)

'''
Winner count:
card_id
Zap               14572
The Log           14475
Fireball          13186
Valkyrie          10658
Knight            10067
Arrows             9404
Hog Rider          9310
Wizard             9172
Skeleton Army      8031
Baby Dragon        7860
Mega Knight        7605
Electro Wizard     7570
Bats               6973
Balloon            6968
Mini P.E.K.K.A     6797
Name: count, dtype: int64 card_id

Loser count:
Zap               13672
The Log           13581
Fireball          12537
Valkyrie          12226
Wizard            11286
Skeleton Army     10178
Arrows            10139
Hog Rider          9660
Knight             9078
Mega Knight        8313
Electro Wizard     7958
Baby Dragon        7741
Mini P.E.K.K.A     7419
Bats               7193
Balloon            6937
'''

# Plot
from matplotlib.ticker import ScalarFormatter

fig, ax = plt.subplots(figsize=(12, 6))

y_pos = range(len(winner_counts))

# Winners (green)
ax.barh([p + 0.2 for p in y_pos], winner_counts.values, height=0.4, color='green', label="Winners")

# Losers (red)
ax.barh([p - 0.2 for p in y_pos], loser_counts.values, height=0.4, color='red', label="Losers")

ax.set_yticks(list(y_pos))
ax.set_yticklabels(winner_counts.index)

ax.set_xlabel("Number of Decks Using Card")
ax.set_title("Top 15 Most Used Cards (Winners vs Losers)")
ax.legend()

# --- Set x-axis to scientific notation ---
ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

plt.tight_layout()
plt.show()


















