import os
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV = os.path.join(ROOT, 'train.csv')
OUT_CSV = os.path.join(ROOT, 'train_features_compared.csv')

print(f"Reading: {TRAIN_CSV}")

df = pd.read_csv(TRAIN_CSV)

# Confirm columns exist
required = ['feature_7', 'feature_10', 'feature_4']
for c in required:
    if c not in df.columns:
        raise SystemExit(f"Required column '{c}' not found in {TRAIN_CSV}")

# Compute sum of feature_7 and feature_10
# Keep original values; create new columns
# Use float to allow NaN
f7 = df['feature_7'].astype(float)
f10 = df['feature_10'].astype(float)
f4 = df['feature_4'].astype(float)

sum_7_10 = f7 + f10

def compare(a, b):
    # Return True only when neither is NaN and they are close numerically
    if pd.isna(a) or pd.isna(b):
        return False
    return float(np.isclose(a, b, atol=1e-9))



equals = [bool(compare(a, b)) for a, b in zip(sum_7_10, f4)]

# Message column
messages = ["feature_7 + feature_10 equals feature_4" if eq else "" for eq in equals]

out_df = df.copy()
out_df['feature_7_plus_10'] = sum_7_10
out_df['equals_feature_4'] = equals
out_df['message'] = messages

# Save to CSV
out_df.to_csv(OUT_CSV, index=False)

# Print summary
count_eq = sum(equals)
count_total = len(out_df)
print(f"Wrote: {OUT_CSV}")
print(f"Total rows: {count_total}")
print(f"Rows where feature_7 + feature_10 equals feature_4: {count_eq}")

# Print some matching rows (up to 10)
if count_eq > 0:
    print('\nSample matching rows (up to 10):')
    sample = out_df.loc[out_df['equals_feature_4']].head(10)
    # Show ID, feature_7, feature_10, feature_4, feature_7_plus_10
    print(sample[['ID', 'feature_7', 'feature_10', 'feature_4', 'feature_7_plus_10', 'message']].to_string(index=False))
else:
    print('No rows matched the equality condition.')


count = df['feature_7'].notnull().sum()
print(f"Total non-missing rows in 'feature_7': {count}")

print("="*70)

# Compute percentage over rows where all three features are present
non_missing_mask = (~f7.isna()) & (~f10.isna()) & (~f4.isna())
denom = int(non_missing_mask.sum())
if denom > 0:
    pct = 100.0 * count_eq / denom
else:
    pct = 0.0

print("="*70)
print(
    f"Percentage of rows (feature_7, feature_10, feature_4 present) where feature_7 + feature_10 == feature_4: {pct:.2f}% ({count_eq}/{denom})"
)