
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
import warnings, json, os
warnings.filterwarnings('ignore')

OUT = '/home/claude/thiranex_project/output/'
os.makedirs(OUT, exist_ok=True)

PALETTE  = ['#0F4C75', '#1B6CA8', '#163172', '#E63946', '#F4A261',
            '#2A9D8F', '#8338EC', '#06D6A0', '#FB8500', '#118AB2']
BG       = '#0D1117'
CARD     = '#161B22'
FG       = '#E6EDF3'
ACCENT   = '#E63946'
GREEN    = '#06D6A0'
BLUE     = '#1B6CA8'

plt.rcParams.update({
    'figure.facecolor':  BG,
    'axes.facecolor':    CARD,
    'axes.edgecolor':    '#30363D',
    'axes.labelcolor':   FG,
    'xtick.color':       FG,
    'ytick.color':       FG,
    'text.color':        FG,
    'grid.color':        '#21262D',
    'grid.linestyle':    '--',
    'grid.alpha':        0.5,
    'font.family':       'DejaVu Sans',
    'legend.facecolor':  CARD,
    'legend.edgecolor':  '#30363D',
})

FSIZE = (14, 7)


print("\n" + "═"*60)
print("  STEP 1  ·  LOAD & INITIAL AUDIT")
print("═"*60)

df_raw = pd.read_csv('/home/claude/thiranex_project/raw_sales_data.csv')
audit = {
    'raw_rows': int(len(df_raw)),
    'raw_cols': int(df_raw.shape[1]),
}

print(f"  Loaded  : {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
print(f"\n  Missing values per column:")
missing = df_raw.isnull().sum()
for col, cnt in missing[missing > 0].items():
    pct = cnt / len(df_raw) * 100
    print(f"    {col:<20} {cnt:>5}  ({pct:.1f}%)")

print(f"\n  Duplicate rows : {df_raw.duplicated().sum():,}")

dtypes_before = df_raw.dtypes.astype(str).to_dict()

print("\n" + "═"*60)
print("  STEP 2  ·  CLEANING PIPELINE")
print("═"*60)

df = df_raw.copy()

df['age'] = pd.to_numeric(df['age'], errors='coerce')
print("  [2-A] Coerced 'age' to numeric — non-numeric strings → NaN")

before_dup = len(df)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
removed_dup = before_dup - len(df)
audit['removed_duplicates'] = removed_dup
print(f"  [2-B] Dropped {removed_dup:,} duplicate rows")

df['purchase_date'] = pd.to_datetime(df['purchase_date'], errors='coerce')
df['returned']      = df['returned'].map({'True': True, 'False': False, True: True, False: False})
print("  [2-C] Parsed 'purchase_date' → datetime, 'returned' → bool")

age_medians = df.groupby('gender')['age'].median()
df['age'] = df.groupby('gender')['age'].transform(
    lambda x: x.fillna(x.median())
)
df['age'].fillna(df['age'].median(), inplace=True)
df['age'] = df['age'].round().astype('Int64')

df['rating'] = df.groupby('product_category')['rating'].transform(
    lambda x: x.fillna(x.median())
)
df['rating'].fillna(df['rating'].median(), inplace=True)

df['purchase_amount'] = df.groupby('product_category')['purchase_amount'].transform(
    lambda x: x.fillna(x.median())
)
df['purchase_amount'].fillna(df['purchase_amount'].median(), inplace=True)
df['gender'].fillna(df['gender'].mode()[0], inplace=True)
df['city'].fillna(df['city'].mode()[0],   inplace=True)

audit['missing_after'] = int(df.isnull().sum().sum())
print(f"  [2-D] Imputed missing values — remaining NaN: {audit['missing_after']}")

Q1, Q3    = df['purchase_amount'].quantile([0.25, 0.75])
IQR       = Q3 - Q1
upper_cap = Q3 + 1.5 * IQR
before_out = len(df)
df_outliers = df[df['purchase_amount'] > upper_cap].copy()   # keep for viz
df = df[df['purchase_amount'] <= upper_cap].reset_index(drop=True)
removed_out = before_out - len(df)
audit['removed_outliers'] = removed_out
audit['outlier_threshold'] = round(upper_cap, 2)
print(f"  [2-E] Removed {removed_out:,} outliers  (purchase_amount > ₹{upper_cap:,.0f})")

df['purchase_month'] = df['purchase_date'].dt.month
df['purchase_month_name'] = df['purchase_date'].dt.strftime('%b')
df['net_amount']     = (df['purchase_amount'] * (1 - df['discount_pct'] / 100)).round(2)
df['age_group']      = pd.cut(df['age'], bins=[17, 25, 35, 45, 55, 100],
                               labels=['18–25', '26–35', '36–45', '46–55', '55+'])
print("  [2-F] Added: purchase_month, net_amount, age_group")

audit['clean_rows'] = len(df)
print(f"\n  ✓ Clean dataset  : {len(df):,} rows × {df.shape[1]} columns")

df.to_csv(OUT + 'cleaned_sales_data.csv', index=False)
print(f"  ✓ Saved → cleaned_sales_data.csv")

print("\n" + "═"*60)
print("  STEP 3  ·  EXPLORATORY STATISTICS")
print("═"*60)

stats = {
    'total_revenue':   round(df['net_amount'].sum(), 2),
    'avg_order_value': round(df['net_amount'].mean(), 2),
    'median_order':    round(df['net_amount'].median(), 2),
    'avg_rating':      round(df['rating'].mean(), 2),
    'return_rate':     round(df['returned'].mean() * 100, 1),
    'top_category':    df['product_category'].value_counts().idxmax(),
    'top_city':        df['city'].value_counts().idxmax(),
}

for k, v in stats.items():
    print(f"  {k:<22} : {v}")

audit.update(stats)
with open(OUT + 'audit_report.json', 'w') as f:
    json.dump(audit, f, indent=2)

def decorate(ax, title, xlabel='', ylabel=''):
    ax.set_title(title, fontsize=14, fontweight='bold', color=FG, pad=14)
    ax.set_xlabel(xlabel, fontsize=11, color=FG)
    ax.set_ylabel(ylabel, fontsize=11, color=FG)
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines[['top','right']].set_visible(False)

print("\n  Generating Figure 1 — Outlier Treatment …")

fig, axes = plt.subplots(1, 2, figsize=FSIZE)
fig.patch.set_facecolor(BG)
fig.suptitle('Distribution of Purchase Amount  ·  Before vs After Cleaning',
             fontsize=15, fontweight='bold', color=FG, y=1.02)

axes[0].hist(df_raw['purchase_amount'].dropna(), bins=60,
             color=ACCENT, edgecolor='none', alpha=0.85)
axes[0].axvline(df_raw['purchase_amount'].median(), color=GREEN,
                lw=2, ls='--', label='Median')
decorate(axes[0], 'RAW  (with outliers)', 'Purchase Amount (₹)', 'Frequency')
axes[0].legend()

axes[1].hist(df['purchase_amount'], bins=60, color=BLUE, edgecolor='none', alpha=0.85)
axes[1].axvline(df['purchase_amount'].median(), color=GREEN,
                lw=2, ls='--', label='Median')
axes[1].axvline(upper_cap, color=ACCENT, lw=2, ls=':', label=f'IQR Cap ₹{upper_cap:,.0f}')
decorate(axes[1], 'CLEANED  (outliers removed)', 'Purchase Amount (₹)', '')
axes[1].legend()

plt.tight_layout()
plt.savefig(OUT + 'fig1_outlier_treatment.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()


print("  Generating Figure 2 — Revenue by Category …")

cat_rev  = df.groupby('product_category')['net_amount'].sum().sort_values()
cat_ret  = df.groupby('product_category')['returned'].mean() * 100

fig, axes = plt.subplots(1, 2, figsize=FSIZE)
fig.patch.set_facecolor(BG)

bars = axes[0].barh(cat_rev.index, cat_rev.values,
                    color=PALETTE[:len(cat_rev)], edgecolor='none', height=0.65)
for bar, val in zip(bars, cat_rev.values):
    axes[0].text(bar.get_width() + cat_rev.max()*0.01,
                 bar.get_y() + bar.get_height()/2,
                 f'₹{val/1e5:.1f}L', va='center', fontsize=9, color=FG)
decorate(axes[0], 'Net Revenue by Category', 'Revenue (₹)', '')
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f'₹{x/1e5:.0f}L'))

colors_ret = [ACCENT if v == cat_ret.max() else '#4A90D9' for v in cat_ret.values]
axes[1].barh(cat_ret.index, cat_ret.values, color=colors_ret, edgecolor='none', height=0.65)
axes[1].axvline(cat_ret.mean(), color=GREEN, lw=1.8, ls='--',
                label=f'Avg {cat_ret.mean():.1f}%')
for i, (idx, val) in enumerate(cat_ret.items()):
    axes[1].text(val + 0.1, i, f'{val:.1f}%', va='center', fontsize=9, color=FG)
decorate(axes[1], 'Return Rate by Category', 'Return Rate (%)', '')
axes[1].legend()

plt.tight_layout()
plt.savefig(OUT + 'fig2_revenue_returns.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

print("  Generating Figure 3 — Monthly Sales Trend …")

month_order = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
monthly = (df.groupby('purchase_month_name')['net_amount']
             .sum()
             .reindex(month_order)
             .fillna(0))
cumulative = monthly.cumsum()

fig, ax1 = plt.subplots(figsize=FSIZE)
fig.patch.set_facecolor(BG)
ax2 = ax1.twinx()

ax1.bar(monthly.index, monthly.values, color=BLUE, alpha=0.75, width=0.6, label='Monthly Revenue')
ax1.fill_between(range(len(monthly)), monthly.values, alpha=0.12, color=BLUE)
ax2.plot(range(len(cumulative)), cumulative.values, color=GREEN,
         lw=2.5, marker='o', ms=6, label='Cumulative Revenue')

ax1.set_xticks(range(len(monthly)))
ax1.set_xticklabels(monthly.index)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x/1e5:.1f}L'))
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x/1e5:.0f}L'))
ax1.set_facecolor(CARD)
ax1.spines[['top']].set_visible(False)
ax2.spines[['top']].set_visible(False)
ax2.tick_params(colors=GREEN)
ax2.yaxis.label.set_color(GREEN)

ax1.set_title('Monthly Revenue Trend  ·  2023', fontsize=14,
              fontweight='bold', color=FG, pad=14)
ax1.set_xlabel('Month', fontsize=11, color=FG)
ax1.set_ylabel('Monthly Revenue (₹)', fontsize=11, color=FG)
ax2.set_ylabel('Cumulative Revenue (₹)', fontsize=11, color=GREEN)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='upper left')
ax1.grid(True, axis='y', alpha=0.25)

plt.tight_layout()
plt.savefig(OUT + 'fig3_monthly_trend.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

print("  Generating Figure 4 — Age Distribution …")

fig, axes = plt.subplots(1, 2, figsize=FSIZE)
fig.patch.set_facecolor(BG)

gender_palette = {'Male': '#1B6CA8', 'Female': '#E63946', 'Non-binary': '#8338EC'}

sns.violinplot(data=df, x='gender', y='age', hue='gender',
               palette=gender_palette, inner='box',
               ax=axes[0], linewidth=1.2, legend=False)
axes[0].set_facecolor(CARD)
decorate(axes[0], 'Age Distribution by Gender', 'Gender', 'Age')

sns.boxplot(data=df, x='age_group', y='purchase_amount', hue='gender',
            palette=gender_palette, linewidth=1.0,
            ax=axes[1], width=0.6)
axes[1].set_facecolor(CARD)
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
decorate(axes[1], 'Purchase Amount by Age Group & Gender', 'Age Group', 'Purchase Amount (₹)')
axes[1].legend(title='Gender', loc='upper right')

plt.tight_layout()
plt.savefig(OUT + 'fig4_age_gender.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

print("  Generating Figure 5 — Correlation Heatmap …")

num_cols = ['age', 'purchase_amount', 'net_amount', 'rating',
            'discount_pct', 'purchase_month']
corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor(BG)
ax.set_facecolor(CARD)

cmap = sns.diverging_palette(220, 10, as_cmap=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
            annot=True, fmt='.2f', linewidths=0.5,
            linecolor='#21262D', square=True,
            ax=ax, annot_kws={'size': 11, 'color': FG},
            cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix', fontsize=14,
             fontweight='bold', color=FG, pad=14)
plt.setp(ax.get_xticklabels(), rotation=35, ha='right', fontsize=10)
plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)

plt.tight_layout()
plt.savefig(OUT + 'fig5_correlation.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

print("  Generating Figure 6 — City Performance …")

city_stats = df.groupby('city').agg(
    orders=('customer_id','count'),
    revenue=('net_amount','sum'),
    avg_rating=('rating','mean'),
    return_rate=('returned','mean')
).reset_index()

fig, ax = plt.subplots(figsize=FSIZE)
fig.patch.set_facecolor(BG)
ax.set_facecolor(CARD)

scatter = ax.scatter(
    city_stats['orders'],
    city_stats['revenue'],
    s=city_stats['avg_rating'] * 400,
    c=city_stats['return_rate'],
    cmap='RdYlGn_r',
    edgecolors=FG, linewidths=0.6,
    alpha=0.85
)
cbar = plt.colorbar(scatter, ax=ax, pad=0.01)
cbar.set_label('Return Rate', color=FG)
cbar.ax.yaxis.set_tick_params(color=FG)
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=FG)

for _, row in city_stats.iterrows():
    ax.annotate(row['city'],
                (row['orders'], row['revenue']),
                xytext=(6, 6), textcoords='offset points',
                fontsize=9, color=FG)

ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x/1e5:.1f}L'))
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x)}'))
decorate(ax, 'City Performance  ·  Orders vs Revenue  (bubble = avg rating, color = return rate)',
         'Order Count', 'Net Revenue (₹)')

plt.tight_layout()
plt.savefig(OUT + 'fig6_city_bubble.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

print("  Generating Figure 7 — Rating & Discount …")

fig, axes = plt.subplots(1, 2, figsize=FSIZE)
fig.patch.set_facecolor(BG)

sns.kdeplot(data=df, x='rating', fill=True, color=BLUE,
            alpha=0.6, linewidth=2, ax=axes[0])
axes[0].axvline(df['rating'].mean(), color=ACCENT, lw=2,
                ls='--', label=f'Mean {df["rating"].mean():.2f}')
axes[0].set_facecolor(CARD)
decorate(axes[0], 'Customer Rating Distribution', 'Rating (1–5)', 'Density')
axes[0].legend()

disc_agg = df.groupby('discount_pct')['net_amount'].mean().reset_index()
axes[1].bar(disc_agg['discount_pct'].astype(str),
            disc_agg['net_amount'],
            color=[GREEN if v < disc_agg['net_amount'].mean() else BLUE
                   for v in disc_agg['net_amount']],
            edgecolor='none')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))
axes[1].set_facecolor(CARD)
decorate(axes[1], 'Avg Net Order Value by Discount %',
         'Discount Applied (%)', 'Avg Net Amount (₹)')

plt.tight_layout()
plt.savefig(OUT + 'fig7_rating_discount.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

print("\n" + "═"*60)
print("  ✓  ALL 7 FIGURES SAVED TO:", OUT)
print("═"*60)
print(f"""
  Summary
  ───────
  Raw rows            : {audit['raw_rows']:>8,}
  Duplicates removed  : {audit['removed_duplicates']:>8,}
  Outliers removed    : {audit['removed_outliers']:>8,}
  Clean rows          : {audit['clean_rows']:>8,}

  Total Revenue       :  ₹{stats['total_revenue']:>12,.2f}
  Avg Order Value     :  ₹{stats['avg_order_value']:>12,.2f}
  Avg Rating          :  {stats['avg_rating']:>8.2f} / 5
  Return Rate         :  {stats['return_rate']:>7.1f} %
  Top Category        :  {stats['top_category']}
  Top City            :  {stats['top_city']}
""")
