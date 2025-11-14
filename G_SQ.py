# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 07:57:36 2025

@author: bishi
"""

import pandas as pd
import numpy as np
from scipy.stats import power_divergence


FILE = "C:\\Users\\bishi\\OneDrive\\Documents\\Python Scripts\\filtered.csv"  
CHUNK_SIZE = 500_000        

CARD_COLS = [1, 3, 5, 7, 9, 11, 13, 15]


CARD_IDS = (
    list(range(26000000, 26000086)) +
    list(range(27000000, 27000013)) +
    list(range(28000000, 28000019))
)
CARD_IDS = set(CARD_IDS)


counts = {card: [0, 0] for card in CARD_IDS}


for chunk in pd.read_csv(FILE, header=None, chunksize=CHUNK_SIZE):
    
    outcome = chunk[0].values  

    cards = chunk[CARD_COLS].values

    for i in range(len(chunk)):
        row_cards = set(cards[i])  
        result = outcome[i]        

        for c in row_cards:
            if c in CARD_IDS:
                if result == 1:
                    counts[c][0] += 1
                else:
                    counts[c][1] += 1


results = []

for card, (wins, losses) in counts.items():

    total_wins = sum(v[0] for v in counts.values())
    total_losses = sum(v[1] for v in counts.values())
    total_uses = wins + losses

    if total_uses == 0:
        continue 

    # Expected values with independence
    expected_wins = total_wins * (total_uses / (total_wins + total_losses))
    expected_losses = total_losses * (total_uses / (total_wins + total_losses))

    observed = np.array([wins, losses])
    expected = np.array([expected_wins, expected_losses])

    # Run G-test
    g_stat, p_value = power_divergence(f_obs=observed,
                                       f_exp=expected,
                                       lambda_="log-likelihood")

    results.append((card, wins, losses, g_stat, p_value))

results_df = pd.DataFrame(
    results,
    columns=["card_id", "wins_with_card", "losses_with_card", "G_statistic", "p_value"]
)

results_df.to_csv("g_test_results.csv", index=False)

print("Done! Results saved to g_test_results.csv")
