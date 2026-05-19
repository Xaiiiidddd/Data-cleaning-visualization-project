# Data-cleaning-visualization-project
Dataset (raw_dataset.py)
A realistic synthetic Customer Sales dataset (1,248 rows) with intentionally injected:

6% missing values across age, gender, city, rating, purchase_amount
~4% duplicate rows (48 records)
30 extreme outliers in purchase_amount (₹80K–₹2L)
String noise in the numeric age column


Cleaning Pipeline (analysis_pipeline.py) — 6-Step Process
StepTechniqueResult2-Apd.to_numeric(errors='coerce')Fixed 15 bad type values2-Bdrop_duplicates()Removed 47 duplicates2-Cpd.to_datetime(), bool mappingFixed 2 column types2-DGroup-aware median imputation by gender/categoryFilled 301 NaNs2-EIQR method (Q3 + 1.5×IQR = ₹7,751 cap)Removed 88 outliers2-FFeature engineeringAdded net_amount, age_group, purchase_month
1,248 raw → 1,113 clean rows

7 Visualizations (embedded in dashboard)

Histogram — Purchase amount before/after outlier treatment
Horizontal bar — Revenue by category + return rate comparison
Dual-axis line+bar — Monthly revenue trend with cumulative overlay
Violin + Boxplot — Age distribution & spend by gender
Correlation heatmap — Pearson matrix of all numeric features
Bubble chart — City performance (orders × revenue × rating × return rate)
KDE + Bar — Rating distribution & discount impact on net value


Dashboard (dashboard.html)
A dark-themed, fully self-contained interactive report with navigation, KPI cards, cleaning pipeline cards, all 7 visualizations, and an 8-finding insights section — ready to submit or present.DashboardCode · HTML DownloadAnalysis pipelinePY DownloadGenerate datasetPY DownloadCleaned sales dataTable · CSV DownloadDownload all
