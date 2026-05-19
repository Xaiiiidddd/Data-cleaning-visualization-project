import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

N = 1200

cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad',
          'Kolkata', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow']
categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Books',
              'Sports', 'Beauty', 'Grocery', 'Toys']
genders = ['Male', 'Female', 'Non-binary']

start_date = datetime(2023, 1, 1)
dates = [start_date + timedelta(days=random.randint(0, 364)) for _ in range(N)]

ages = np.random.normal(35, 12, N).astype(int)
ages = np.clip(ages, 18, 80)

purchase_amounts = np.random.exponential(scale=2500, size=N)
purchase_amounts = np.clip(purchase_amounts, 50, 50000)

outlier_idx = np.random.choice(N, size=30, replace=False)
purchase_amounts[outlier_idx] = np.random.uniform(80000, 200000, size=30)

ratings = np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                            size=N, p=[0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.25, 0.18, 0.12])

discounts = np.random.choice([0, 5, 10, 15, 20, 25, 30], size=N,
                              p=[0.3, 0.1, 0.2, 0.15, 0.12, 0.08, 0.05])

df = pd.DataFrame({
    'customer_id':        [f'CUST{str(i).zfill(5)}' for i in range(1, N+1)],
    'age':                ages,
    'gender':             np.random.choice(genders, N, p=[0.52, 0.44, 0.04]),
    'city':               np.random.choice(cities, N),
    'product_category':   np.random.choice(categories, N),
    'purchase_amount':    purchase_amounts.round(2),
    'purchase_date':      dates,
    'rating':             ratings,
    'discount_pct':       discounts,
    'returned':           np.random.choice([True, False], N, p=[0.12, 0.88]),
})

for col, frac in [('age', 0.06), ('rating', 0.09), ('purchase_amount', 0.04),
                   ('gender', 0.03), ('city', 0.02)]:
    miss_idx = np.random.choice(N, size=int(N * frac), replace=False)
    df.loc[miss_idx, col] = np.nan

dup_rows = df.sample(frac=0.04, random_state=7)
df = pd.concat([df, dup_rows], ignore_index=True)
df = df.sample(frac=1, random_state=99).reset_index(drop=True)

bad_idx = np.random.choice(len(df), size=15, replace=False)
df['age'] = df['age'].astype(object)
df.loc[bad_idx, 'age'] = 'unknown'

df.to_csv('/home/claude/thiranex_project/raw_sales_data.csv', index=False)
print(f"Raw dataset saved — shape: {df.shape}")
