#---------------------
# 4.1 Data Preparation
#------------------------------------
# 4.1.1.1 Merging all the UN datasets
#------------------------------------
import csv
import os
import glob
from collections import defaultdict

folder_path = os.path.expanduser("~/Desktop/FMS/UN Trade Datasets")
file_list = glob.glob(os.path.join(folder_path, "*.csv"))
merged_data = defaultdict(lambda: [0]*len(file_list))

# Loop through each file
for idx, file in enumerate(file_list):
    filename = os.path.splitext(os.path.basename(file))[0]
    
    # Attempt UTF-8, fallback to ISO-8859-1
    for encoding in ['utf-8', 'ISO-8859-1']:
        try:
            with open(file, newline='', encoding=encoding) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    
                    # Standardise keys
                    refYear = row['refYear'].strip()
                    reporterDesc = row['reporterDesc'].strip().upper()
                    flowDesc = row['flowDesc'].strip().upper()
                    key = (refYear, reporterDesc, flowDesc)

                    # Convert primaryValue to numeric
                    try:
                        value = float(row['primaryValue'].replace(',', ''))
                    except:
                        value = 0

                    # Sum duplicates
                    merged_data[key][idx] += value
                    
            break  
        except UnicodeDecodeError:
            continue  

# Establish header using filenames
header = ['refYear', 'reporterDesc', 'flowDesc'] + [os.path.splitext(os.path.basename(f))[0] for f in file_list]

# Write merged CSV
output_file = os.path.join(folder_path, "merged_dataset_clean.csv")
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for key, values in merged_data.items():
        writer.writerow(list(key) + values)

print("Clean merged dataset saved as 'merged_dataset_clean.csv'!")


#-----------
#Sum columns
#-----------
import pandas as pd

# Load merged dataset
df = pd.read_csv("~/Desktop/FMS/UN Trade Datasets/merged_dataset_clean.csv")

# Identify the 10 trade value columns
value_columns = df.columns[3:]

# Create new column that sums all 10 datasets for each row
df['total_trade'] = df[value_columns].sum(axis=1)

# Keep only the needed columns
df_final = df[['refYear', 'reporterDesc', 'flowDesc', 'total_trade']]

# Group by year, country, and flowDesc to sum across all types of apparel trade
df_final = df_final.groupby(['refYear', 'reporterDesc', 'flowDesc'], as_index=False).sum()

# Save the final dataset
df_final.to_csv("~/Desktop/FMS/UN Trade Datasets/merged_dataset_final.csv", index=False)

print("Final dataset saved as 'merged_dataset_final.csv'!")



#--------------------------------
# Calculate import/export balance
#--------------------------------
import pandas as pd
import numpy as np
import os

# Load final dataset
df = pd.read_csv(os.path.expanduser("~/Desktop/FMS/UN Trade Datasets/merged_dataset_final.csv"))

# Normalise flowDesc 
df['flowDesc'] = df['flowDesc'].str.strip().str.lower()

# Pivot so that import and export become separate columns
df_pivot = df.pivot_table(index=['refYear', 'reporterDesc'],
                          columns='flowDesc',
                          values='total_trade',
                          aggfunc='sum',
                          fill_value=np.nan
                          ).reset_index()

# Rename columns consistently
rename_map = {}
if 'import' in df_pivot.columns:
    rename_map['import'] = 'Import'
if 'imports' in df_pivot.columns:
    rename_map['imports'] = 'Import'
if 'export' in df_pivot.columns:
    rename_map['export'] = 'Export'
if 'exports' in df_pivot.columns:
    rename_map['exports'] = 'Export'

df_pivot = df_pivot.rename(columns=rename_map)

# Calculate trade balance
df_pivot['Trade Balance'] = df_pivot['Export'] - df_pivot['Import']

# Rename refYear and reporterDesc to final column names
df_pivot = df_pivot.rename(columns={
    'refYear': 'Year',
    'reporterDesc': 'Country'
})

# Reorder columns
final_df = df_pivot[['Year', 'Country', 'Import', 'Export', 'Trade Balance']]

# Save final dataset
output_file = os.path.expanduser("~/Desktop/FMS/UN Trade Datasets/Final_UN_dataset.csv")
final_df.to_csv(output_file, index=False)

print(f"Trade balance dataset saved as '{output_file}'!")





#--------------------------------------------
# 4.1.1.2 Merging with the World Bank Dataset 
#--------------------------------------------
# Cleaning the World Bank dataset
#--------------------------------
import pandas as pd
import os

# File path
file_path = os.path.expanduser("~/Desktop/FMS/World_bank_dataset.csv")

# Read the file and skip the first 4 rows
df = pd.read_csv(file_path, skiprows=4, header=0)

# Save as a new dataset
output_file = os.path.expanduser("~/Desktop/FMS/World_bank_dataset_cleaned.csv")
df.to_csv(output_file, index=False)

print(f"Cleaned dataset saved as 'World_bank_dataset_cleaned.csv'")



#-------------------------------------
# Merge all into the final FMS dataset
#-------------------------------------
import pandas as pd
import os

# File paths
trade_file = os.path.expanduser("~/Desktop/FMS/Final_UN_dataset.csv")
wb_file = os.path.expanduser("~/Desktop/FMS/World_bank_dataset_cleaned.csv")

# Load trade dataset
trade_df = pd.read_csv(trade_file)

# Normalise country names for merging
trade_df['Country_upper'] = trade_df['Country'].str.upper()
trade_df['Year'] = trade_df['Year'].astype(int)

# Load World Bank dataset
wb_df = pd.read_csv(wb_file)

# Normalise country names
wb_df['Country_upper'] = wb_df['Country Name'].str.upper()

# Melt World Bank dataset so years become rows
years = [str(y) for y in range(1996, 2024)]  # World Bank covers 1996–2023
wb_melted = wb_df.melt(id_vars=['Country_upper'],
                       value_vars=years,
                       var_name='Year',
                       value_name='Regulatory Quality')

# Convert year to numeric
wb_melted['Year'] = wb_melted['Year'].astype(int)

# Merge datasets
merged_full = pd.merge(
    trade_df,
    wb_melted,
    left_on=['Country_upper', 'Year'],
    right_on=['Country_upper', 'Year'],
    how='outer'
)

# Fill in Country name consistently
merged_full['Country'] = merged_full['Country'].fillna(merged_full['Country_upper'])

# Rename imports/exports
merged_full = merged_full.rename(columns={
    'Import': 'Total Import',
    'Export': 'Total Export'
})

# Keep and reorder final columns
final_df = merged_full[['Year', 'Country', 'Total Import', 'Total Export', 'Trade Balance', 'Regulatory Quality']]

# Drop rows where all relevant columns are NaN
final_df = final_df.dropna(subset=['Total Import', 'Total Export', 'Trade Balance', 'Regulatory Quality'], how='all')

# Save final dataset
output_file = os.path.expanduser("~/Desktop/FMS/FMS_dataset.csv")
final_df.to_csv(output_file, index=False)

print(f"Final dataset saved as '{output_file}'")


#----------------------------------------
# 4.1.1.3 Data cleaning & transformations
#----------------------------------------
import pandas as pd
import os

# Load merged dataset
file_path = os.path.expanduser("~/Desktop/FMS/FMS_dataset.csv")
df = pd.read_csv(file_path)

# Standardise country names
country_map = {
    'BAHAMAS, THE': 'BAHAMAS',
    'BOLIVIA (PLURINATIONAL STATE OF)':'BOLIVIA',
    'BOSNIA HERZEGOVINA':'BOSNIA AND HERZEGOVINA',

    'CAYMAN ISDS': 'CAYMAN ISLANDS',
    'CENTRAL AFRICAN REP.': 'CENTRAL AFRICAN REPUBLIC',
    'CHINA, HONG KONG SAR': 'HONG KONG',
    'HONG KONG SAR, CHINA': 'HONG KONG',
    'CHINA, MACAO SAR': 'MACAO',
    'MACAO SAR, CHINA': 'MACAO',
    'CONGO, REP.': 'CONGO',
    'DEM. REP. OF THE CONGO': 'DEMOCRATIC REPUBLIC OF THE CONGO',
    'CONGO, DEM. REP.': 'DEMOCRATIC REPUBLIC OF THE CONGO',
    "CÔTE D'IVOIRE": "COTE D'IVOIRE",
    'DOMINICAN REP.': 'DOMINICAN REPUBLIC',
    'EGYPT, ARAB REP.': 'EGYPT',
    'GAMBIA, THE': 'GAMBIA',
    'IRAN, ISLAMIC REP.': 'IRAN',
    'REP. OF KOREA': 'KOREA, REP.',
    'KYRGYZSTAN': 'KYRGYZ REPUBLIC',
    "LAO PEOPLE'S DEM. REP.": 'LAO PDR',
    'REP. OF MOLDOVA': 'MOLDOVA',
    'ST. KITTS AND NEVIS': 'SAINT KITTS AND NEVIS',
    'ST. LUCIA': 'SAINT LUCIA',
    'ST. VINCENT AND THE GRENADINES': 'SAINT VINCENT AND THE GRENADINES',
    'SLOVAK REPUBLIC': 'SLOVAKIA',
    'SOLOMON ISDS': 'SOLOMON ISLANDS',
    'UNITED REP. OF TANZANIA': 'TANZANIA',
    'TÜRKIYE': 'TURKIYE',
    'USA': 'UNITED STATES',
    'WEST BANK AND GAZA': 'PALESTINE',
    'STATE OF PALESTINE': 'PALESTINE'
}

df['Country'] = df['Country'].replace(country_map)

# Remove countries with insufficient data
remove_countries = [
    'ALGERIA','BANGLADESH','BHUTAN','CHAD','CURAÇAO', 'DJIBOUTI', 'ERITREA', 'EQUATORIAL GUINEA','FRENCH POLYNESIA', 'FS MICRONESIA',
    'GREENLAND','GUAM', 'GUINEA', 'GUINEA-BISSAU', 'HAITI', 'IRAQ', "KOREA, DEM. PEOPLE'S REP.",
    'KOSOVO', 'LIBYA', 'LIECHTENSTEIN', 'MALI', 'MARSHALL ISLANDS',
    'MICRONESIA, FED. STS.', 'MONACO', 'MONTSERRAT', 'NAURU', 'NEW CALEDONIA',
    'OTHER ASIA, NES', 'PALAU','PAPUA NEW GUINEA', 'PUERTO RICO (US)', 'SAN MARINO',
    'SOMALIA', 'SOUTH SUDAN', 'SAINT KITTS AND NEVIS', 'SIERRA LEONE','SOLOMON ISLANDS','SYRIAN ARAB REPUBLIC',
    'SUDAN','TONGA', 'TURKMENISTAN', 'TUVALU', 'VANUATU', 'VENEZUELA, RB', 'VIRGIN ISLANDS (U.S.)', 'YEMEN', 'YEMEN, REP.'
]

df = df[~df['Country'].isin(remove_countries)]

# Merge duplicates 
numeric_cols = ['Total Import', 'Total Export', 'Trade Balance', 'Regulatory Quality']
df_clean = df.groupby(['Country', 'Year'], as_index=False)[numeric_cols].sum(min_count=1)

# For rows where Trade Balance is NaN
mask = df_clean['Trade Balance'].isna()

# If Total Export is missing but Total Import exists, Trade Balance = - Total Import
df_clean.loc[mask & df_clean['Total Import'].notna(), 'Trade Balance'] = -df_clean.loc[mask & df_clean['Total Import'].notna(), 'Total Import']

# If Total Import is missing but Total Export exists, Trade Balance = Total Export
df_clean.loc[mask & df_clean['Total Export'].notna(), 'Trade Balance'] = df_clean.loc[mask & df_clean['Total Export'].notna(), 'Total Export']

# Sort by Country, then Year ascending
df_clean = df_clean.sort_values(['Country', 'Year']).reset_index(drop=True)

# Save cleaned dataset 
output_file = os.path.expanduser("~/Desktop/FMS/FMS_dataset_clean.csv")
df_clean.to_csv(output_file, index=False)

print(f" Clean dataset saved as '{output_file}'")




#---------------------
# Do I even need this?
#--------------------------
# 4.2 Descriptive Analysis
#--------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = os.path.expanduser("~/Desktop/FMS/FMS_dataset_clean.csv")
df['Country'] = df['Country'].astype(str).str.strip()

# Aggregate by country according to mean over years
df_agg = df.groupby('Country', as_index=False).agg({
    'Trade Balance': 'mean',
    'Regulatory Quality': 'mean'
})
df_agg['Trade Balance (Billion USD)'] = df_agg['Trade Balance'] / 1e9

# Compute averages
avg_trade = df_agg['Trade Balance (Billion USD)'].mean()
avg_reg_quality = df_agg['Regulatory Quality'].mean()

# Identify top 5 and bottom 5 countries by Trade Balance
top_bottom = pd.concat([
    df_agg.nlargest(5, 'Trade Balance (Billion USD)'),
    df_agg.nsmallest(5, 'Trade Balance (Billion USD)')
]).reset_index(drop=True)

# Prepare colours for highlighted countries
palette = sns.color_palette('tab10', n_colors=len(top_bottom))

plt.figure(figsize=(14,10))

# Plot all countries in gray
sns.scatterplot(
    data=df_agg,
    x='Regulatory Quality',
    y='Trade Balance (Billion USD)',
    color='lightgray',
    s=100
)

# Highlight top/bottom 5 in different colors
for i, row in top_bottom.iterrows():
    plt.scatter(
        row['Regulatory Quality'],
        row['Trade Balance (Billion USD)'],
        color=palette[i],
        s=150
    )
    plt.text(
        row['Regulatory Quality'] + 0.5,
        row['Trade Balance (Billion USD)'] + 10,
        row['Country'],
        fontsize=10,
        weight='bold')
    
# Convert Trade Balance to billions for readability
df['Trade Balance (Billion USD)'] = df['Trade Balance'] / 1e9

# Compute yearly averages
yearly_avg = df.groupby('Year').agg({
    'Trade Balance (Billion USD)': 'mean',
    'Regulatory Quality': 'mean'
}).reset_index()

# Plot evolution over years
plt.figure(figsize=(12,7))
sns.lineplot(data=yearly_avg, x='Year', y='Trade Balance (Billion USD)', marker='o', label='Trade Balance')
sns.lineplot(data=yearly_avg, x='Year', y='Regulatory Quality', marker='o', label='Regulatory Quality (%)')

plt.title('Global Average Trade Balance and Regulatory Quality Over Time')
plt.xlabel('Year')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()





#-------------------------
#4.3.1 Regression Analysis
#-------------------------
#Save dataset for regression
#--------------------------
import pandas as pd
import os

# Load the dataset
file_path = os.path.expanduser("~/Desktop/FMS/FMS_dataset_clean.csv")
df = pd.read_csv(file_path)

# Keep only relevant columns
df = df[['Country', 'Year', 'Trade Balance', 'Regulatory Quality']].copy()

# Remove extreme outliers
df = df[~df['Country'].isin(['CHINA', 'UNITED STATES'])]

# Drop rows with missing values
df = df.dropna(subset=['Trade Balance', 'Regulatory Quality'])

# Save to new CSV
output_path = os.path.expanduser("~/Desktop/FMS/FMS_regression.csv")
df.to_csv(output_path, index=False)

print(f"Filtered dataset saved to: {output_path}")


#--------------------
# Model specification
#--------------------
# Testing
#--------
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro
import os


# Load and clean dataset
file_path = os.path.expanduser("~/Desktop/FMS/FMS_regression.csv")
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()  # strip spaces

# Set panel index
df = df.set_index(['Country', 'Year'])

# Create within- and between-country components
# Country mean
df['Reg_Q_Mean'] = df.groupby('Country')['Regulatory Quality'].transform('mean')

# Within-country deviation
df['Reg_Q_Within'] = df['Regulatory Quality'] - df['Reg_Q_Mean']

# Dependent variable
y = df['Trade Balance']

# Independent variables
X = df[['Reg_Q_Within', 'Reg_Q_Mean']]
X = sm.add_constant(X)

# Preliminary tests
print("Correlation matrix:")
print(df[['Trade Balance','Reg_Q_Within','Reg_Q_Mean']].corr())

# Pooled OLS residuals for diagnostics
pooled_model = sm.OLS(y, X).fit()
residuals = pooled_model.resid

# Shapiro-Wilk test of normality
shapiro_test = shapiro(residuals)
print("\nShapiro-Wilk test:", shapiro_test)

# Breusch-Pagan test of heteroskedasticity
bp_test = het_breuschpagan(residuals, X)
labels = ['LM Statistic','LM p-value','F-Statistic','F-Test p-value']
print("\nBreusch-Pagan test:", dict(zip(labels, bp_test)))

# Durbin-Watson test of serial correlation
dw_stat = durbin_watson(residuals)
print("\nDurbin-Watson statistic:", dw_stat)

# Variance Inflation Factor
vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nVariance Inflation Factor (VIF):")
print(vif_data)

# Run within-between regression
model = sm.OLS(y, X).fit(cov_type='HC3')  # robust SE
print("\n----- Within-Between (Hybrid) Regression Results -----")
print(model.summary())

# Within-Between Regression with Clustered SEs
# Re-fit the model
model_clustered = sm.OLS(y, X).fit(
    cov_type='cluster',
    cov_kwds={'groups': df.index.get_level_values("Country")}
)

print("\n----- Within-Between (Hybrid) Regression with Clustered SEs -----")
print(model_clustered.summary())

# --- PURE WITHIN (Fixed-Effects style) ---
# Demean by country: y_it - y_bar_i, x_it - x_bar_i
df_demeaned = df.copy()
df_demeaned['y_demeaned'] = df.groupby('Country')['Trade Balance'].transform(lambda x: x - x.mean())
df_demeaned['x_demeaned'] = df.groupby('Country')['Regulatory Quality'].transform(lambda x: x - x.mean())

y_w = df_demeaned['y_demeaned']
X_w = sm.add_constant(df_demeaned['x_demeaned'])
within_model = sm.OLS(y_w, X_w).fit(cov_type='cluster',
    cov_kwds={'groups': df.index.get_level_values("Country")})
print("\n----- Within Regression with Fixed Effects -----")
print(within_model.summary())

# --- PURE BETWEEN (Country averages) ---
df_between = df.groupby("Country").agg({
    "Trade Balance": "mean",
    "Regulatory Quality": "mean"
}).reset_index()

y_b = df_between["Trade Balance"]
X_b = sm.add_constant(df_between["Regulatory Quality"])
between_model = sm.OLS(y_b, X_b).fit()
print("\n----- Between Regression based on Country Averages -----")
print(between_model.summary())

# Histogram + KDE of residuals
plt.figure(figsize=(8,5))
sns.histplot(residuals, kde=True, bins=30, color='skyblue')
plt.title("Residuals Distribution")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

# Q-Q plot
plt.figure(figsize=(6,6))
sm.qqplot(residuals, line='45', fit=True)
plt.title("Q-Q Plot of Residuals")
plt.show()









#----------------------------------
# Normalisation & Re-run Regression
#----------------------------------
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro

# Make a copy to preserve original
df_norm = df.copy()

# Log-transform (shift if negative)
if min_tb <= 0:
    df_norm['Trade_B_log'] = np.log(df_norm['Trade Balance'] - min_tb + 1)
else:
    df_norm['Trade_B_log'] = np.log(df_norm['Trade Balance'])

# Dependent variable
y_log = df_norm['Trade_B_log']

# Independent variables (same as before)
X = df_norm[['Reg_Q_Within', 'Reg_Q_Mean']]
X = sm.add_constant(X)

# OLS with robust SE
model_log = sm.OLS(y_log, X).fit(cov_type='HC3')
print("\n----- Log-Transformed Regression (HC3) -----")
print(model_log.summary())

# Clustered SEs by Country
model_log_cluster = sm.OLS(y_log, X).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_norm.index.get_level_values("Country")}
)
print("\n----- Log-Transformed Regression (Clustered SEs) -----")
print(model_log_cluster.summary())

# Shapiro-Wilk test of normality
residuals_log = model_log.resid
shapiro_test_log = shapiro(residuals_log)
print("\nShapiro-Wilk test (log-transformed residuals):", shapiro_test_log)

# Breusch-Pagan test of heteroskedasticity
bp_test_log = het_breuschpagan(residuals_log, X)
labels = ['LM Statistic','LM p-value','F-Statistic','F-Test p-value']
print("\nBreusch-Pagan test (log-transformed residuals):", dict(zip(labels, bp_test_log)))

# Durbin-Watson test of serial correlation
dw_stat_log = durbin_watson(residuals_log)
print("\nDurbin-Watson statistic (log-transformed residuals):", dw_stat_log)

# Residuals from log-transformed model
residuals_log = model_log.resid  # or model_log_cluster.resid if you prefer clustered

# Histogram + KDE
plt.figure(figsize=(8,5))
sns.histplot(residuals_log, kde=True, bins=30, color='skyblue')
plt.title("Residuals Distribution (Log-Transformed Trade Balance)")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

# Q-Q plot
plt.figure(figsize=(6,6))
sm.qqplot(residuals_log, line='45', fit=True)
plt.title("Q-Q Plot of Residuals (Log-Transformed Trade Balance)")
plt.show()






#------------
# Add Lag
#------------
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro

# Load and clean dataset
file_path = os.path.expanduser("~/Desktop/FMS/FMS_regression.csv")
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()  

# Set panel index
df = df.set_index(['Country', 'Year'])

# Remove obvious outliers
df_no_outliers = df.copy()  

# Confirm within- and between-country components exist in df_no_outliers
df_no_outliers['Reg_Q_Mean'] = df_no_outliers.groupby('Country')['Regulatory Quality'].transform('mean')
df_no_outliers['Reg_Q_Within'] = df_no_outliers['Regulatory Quality'] - df_no_outliers['Reg_Q_Mean']

# Create lagged within-country variable
df_no_outliers['Reg_Q_Within_Lag1'] = df_no_outliers.groupby('Country')['Reg_Q_Within'].shift(1)
df_no_outliers['Reg_Q_Within_Lag2'] = df_no_outliers.groupby('Country')['Reg_Q_Within'].shift(2)


# Drop rows with NaNs created by lag or any existing missing data in predictors
df_lagged = df_no_outliers.dropna(subset=['Reg_Q_Within', 'Reg_Q_Within_Lag1', 'Reg_Q_Within_Lag2', 'Reg_Q_Mean', 'Trade Balance'])

# Set up dependent and independent variables
y = df_lagged['Trade Balance']
X = df_lagged[['Reg_Q_Within', 'Reg_Q_Within_Lag1', 'Reg_Q_Within_Lag2', 'Reg_Q_Mean']]
X = sm.add_constant(X)

# Run OLS regression with clustered SEs
model_clustered = sm.OLS(y, X).fit(
    cov_type='cluster',
    cov_kwds={'groups': df_lagged.index.get_level_values('Country')}
)

print(model_clustered.summary())

# Residual diagnostics
residuals = model_clustered.resid

# Shapiro-Wilk test of normality
shapiro_test = shapiro(residuals)
print("\nShapiro-Wilk test:", shapiro_test)

# Histogram + KDE
sns.histplot(residuals, kde=True, bins=30)
plt.title("Residuals Distribution with Lags")
plt.show()

# Q-Q plot
sm.qqplot(residuals, line='45')
plt.title("Q-Q Plot of Residuals with Lags")
plt.show()

# Skewness and kurtosis
print("Skew:", residuals.skew())
print("Kurtosis:", residuals.kurtosis())














#-------------------------------
# 4.3.1 Country-Level Clustering 
#-------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
file_path = os.path.expanduser("~/Desktop/FMS/FMS_dataset_clean.csv")
df = pd.read_csv(file_path)

# Strip column spaces
df.columns = df.columns.str.strip()

# Keep relevant columns and drop missing/infinite values
df = df[['Country','Year','Trade Balance','Regulatory Quality']].replace([np.inf,-np.inf],np.nan)
df = df.dropna(subset=['Trade Balance','Regulatory Quality'])

# Aggregate by country 
df_country = df.groupby('Country')[['Trade Balance','Regulatory Quality']].mean().reset_index()

# Define quantile thresholds
trade_median = df_country['Trade Balance'].median()
reg_median = df_country['Regulatory Quality'].median()

print("Trade Balance median:", trade_median)
print("Regulatory Quality median:", reg_median)

# Assign clusters based on quantiles
def assign_cluster(row):
    if row['Trade Balance'] < trade_median and row['Regulatory Quality'] < reg_median:
        return 'Low Reg, Low Trade'
    elif row['Trade Balance'] >= trade_median and row['Regulatory Quality'] < reg_median:
        return 'Low Reg, High Trade'
    elif row['Trade Balance'] >= trade_median and row['Regulatory Quality'] >= reg_median:
        return 'High Reg, High Trade'
    elif row['Trade Balance'] < trade_median and row['Regulatory Quality'] >= reg_median:
        return 'High Reg, Low Trade'
    else:
        return 'Other'

df_country['Cluster_Label'] = df_country.apply(assign_cluster, axis=1)

# Visualisation
plt.figure(figsize=(14,10))
palette = {
    'High Reg, High Trade': '#1f77b4',
    'High Reg, Low Trade': '#ff7f0e',
    'Low Reg, High Trade': '#2ca02c',
    'Low Reg, Low Trade': '#d62728',
    'Other': '#7f7f7f'
}

sns.scatterplot(data=df_country, 
                x='Regulatory Quality', 
                y='Trade Balance', 
                hue='Cluster_Label',
                palette=palette,
                s=120,
                legend='full')

plt.title('Country-Level Clustering by Regulatory Quality & Trade Balance')
plt.xlabel('Regulatory Quality')
plt.ylabel('Trade Balance')
plt.legend(title='Cluster Group')
plt.show()

# Count of countries per cluster
cluster_counts = df_country['Cluster_Label'].value_counts()
print("Number of countries per cluster:\n")
print(cluster_counts)

# Countries in each cluster
print("\nCountries per cluster:\n")
for cluster_name, group in df_country.groupby('Cluster_Label'):
    print(f"{cluster_name} ({len(group)} countries):")
    print(", ".join(group['Country'].tolist()))

# Identify Key Countries per Cluster
key_countries = {}

for cluster in ['Low Reg, High Trade', 'High Reg, Low Trade', 'High Reg, High Trade', 'Low Reg, Low Trade']:
    subset = df_country[df_country['Cluster_Label'] == cluster]
    if not subset.empty:
        if 'Low Reg' in cluster:
            reg_country = subset.loc[subset['Regulatory Quality'].idxmin()]['Country']
        else:
            reg_country = subset.loc[subset['Regulatory Quality'].idxmax()]['Country']
        if 'Low Trade' in cluster:
            trade_country = subset.loc[subset['Trade Balance'].idxmin()]['Country']
        else:
            trade_country = subset.loc[subset['Trade Balance'].idxmax()]['Country']
        key_countries[cluster] = {
            'Representative Regulatory Quality Country': reg_country,
            'Representative Trade Balance Country': trade_country
        }

# Print results
print("\nKey Representative Countries by Cluster:\n")
for cluster, highlights in key_countries.items():
    print(f"{cluster}:")
    for k, v in highlights.items():
        print(f"  - {k}: {v}")
    print()

# Export results for Tableau
output_path = os.path.expanduser("~/Desktop/FMS/FMS_clusters.csv")
df_country.to_csv(output_path, index=False)
print(f"\nClustered dataset exported to: {output_path}")
