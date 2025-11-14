# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 12:41:49 2025

@author: David
"""

import pandas as pd
import os
import glob

# 1. Read data in chunks
# -----------------------------
chunksize = 10000
file_path = "C:/Users/David/Documents/Fall 2025/datathon/battles.csv" 
chunks = []

# Output folder for chunked Parquet files
output_dir = r"C:\Users\David\Documents\Fall 2025\datathon\ClashRoyale_parquet_parts"
os.makedirs(output_dir, exist_ok=True)

# Read and convert in chunks
chunksize = 200_000
i = 0

for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):
    print(f"Processing chunk {i}...")
    chunk.to_parquet(
        os.path.join(output_dir, f"ClashRoyale_part{i}.parquet"),
        compression="snappy"
    )
    i += 1

print("\n Chunk conversion complete!")

# Combine all Parquet parts into one file
print("Combining into single Parquet file...")
files = sorted(glob.glob(os.path.join(output_dir, "*.parquet")))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
df.to_parquet(r"C:\Users\David\Documents\Fall 2025\datathon\ClashRoyale_full.parquet", compression="snappy")

print("\nDone! Saved as ClashRoyale_full.parquet")

for chunk in pd.read_csv(file_path, chunksize=chunksize):
    chunk = chunk[chunk['gameMode.id'] == 1]  # filter game mode if desired
    chunks.append(chunk)

battles = pd.concat(chunks, ignore_index=True)

battles = pd.read_csv(file_path)

battles.to_csv("battles_sample.csv", index=False) 

#%% here  renamed it with the named card id's
# File path
file_path = "C:/Users/David/Documents/Fall 2025/datathon/battles_sample_named.csv"

# Load the dataset (parse battleTime as datetime)
battles = pd.read_csv(file_path, parse_dates=['battleTime'])

# Filter by specific dates
mask_1209 = (battles['battleTime'].dt.date == pd.to_datetime('2020-12-09').date())
battles_12_09 = battles.loc[mask_1209]

# Save each filtered dataset to separate CSV files
battles_12_09.to_csv("C:/Users/David/Documents/Fall 2025/datathon/battles_2020_12_09.csv", index=False)

print("Files saved successfully:")
print(" - battles_2020_12_09.csv")





