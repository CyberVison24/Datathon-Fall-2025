# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 18:26:49 2025

@author: bishi
"""

import csv
from collections import defaultdict
from scipy.stats import fisher_exact
import numpy as np
import matplotlib.pyplot as plt

input_file = "C:\\Users\\bishi\\OneDrive\\Documents\\Python Scripts\\filtered.csv"

# Columns to check for card presence (0-indexed)
card_columns = [1, 3, 5, 7, 9, 11, 13, 15]

# Count card usage in winners and losers
card_counts = defaultdict(lambda: [0, 0])  # [winners_present, losers_present]
total_winners = 0
total_losers = 0

with open(input_file, "r", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < max(card_columns) + 1:
            continue

        winner_flag = int(row[0])
        if winner_flag == 1:
            total_winners += 1
        else:
            total_losers += 1

        cards_in_row = set([v for i, v in enumerate(row) if i in card_columns and v.strip() != "" and v.strip() != "0"])

        for card in cards_in_row:
            if winner_flag == 1:
                card_counts[card][0] += 1
            else:
                card_counts[card][1] += 1

# Fisher's Exact Test
p_values = []
card_list = sorted(card_counts.keys())  # sorted list of cards
results = {}

for card in card_list:
    w_present, l_present = card_counts[card]
    w_absent = total_winners - w_present
    l_absent = total_losers - l_present

    table = [[w_present, w_absent],
             [l_present, l_absent]]

    odds_ratio, p_value = fisher_exact(table)
    p_values.append(p_value)
    results[card] = {
        "w_present": w_present,
        "l_present": l_present,
        "odds_ratio": odds_ratio,
        "p_value": p_value
    }

# Print results
for card, stats in results.items():
    print(f"Card {card}: winners={stats['w_present']}, losers={stats['l_present']}, "
          f"odds_ratio={stats['odds_ratio']:.3f}, p_value={stats['p_value']:.6f}")

print("\nVector of p-values:")
print(p_values)

sorted_cards = sorted(results.keys(), key=lambda c: results[c]['odds_ratio'], reverse=True)


top_n = 15
top_cards = sorted_cards[:top_n]
odds_ratios = [results[c]['odds_ratio'] for c in top_cards]

# Plot
x = np.arange(len(top_cards))
fig, ax = plt.subplots(figsize=(12,6))
bars = ax.bar(x, odds_ratios, color='skyblue')

ax.axhline(1, color='red', linestyle='--', linewidth=1)

# Labels
ax.set_xticks(x)
ax.set_xticklabels(top_cards, rotation=45)
ax.set_ylabel("Odds Ratio (Winner vs Loser)")
ax.set_title(f"Top {top_n} Cards by Odds Ratio")
plt.tight_layout()
plt.show()



# Graph
plt.figure(figsize=(4, 1))  # short height for a number line
plt.scatter(odds_ratios, [0]*len(odds_ratios), color='blue', s=100)  # y=0 for all points
plt.hlines(0, min(vector)-1, max(vector)+1, color='black', linewidth=1)

# Labels
plt.yticks([])
plt.xlabel("Value")
plt.title("Number Line Plot")
plt.xlim(min(vector)-1, max(vector)+1)

plt.show()