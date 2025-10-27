import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # Ensure plt switch backend for plotting

import seaborn as sns
sns.set(style='whitegrid')

# For inline plotting in Jupyter Notebook
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

print('Libraries imported and warnings suppressed')
data_path = '/kaggle/input/coca-cola-stock-daily-updated/Coca_Cola_historical_data.csv'
df = pd.read_csv(data_path, encoding='ascii', delimiter=',')

# Display the first few rows of the dataframe
print('Data loaded successfully:')
print(df.head())

# Check the DataFrame's info
print('\nDataFrame Info:')
print(df.info())
# Convert 'Date' column to datetime format
# This is important because date-based operations require datetime objects
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Check for missing or corrupted date entries
missing_dates = df['Date'].isna().sum()
if missing_dates > 0:
    print(f'Found {missing_dates} missing or invalid date entries. Consider handling them appropriately.')
else:
    print('All dates converted successfully.')

# Optionally, sort the DataFrame by Date if time series continuity matters
df = df.sort_values('Date').reset_index(drop=True)

# Check for missing values in other columns
missing_values = df.isnull().sum()
print('\nMissing values in each column:')
print(missing_values)

# Create a copy of the dataframe for EDA purposes
eda_df = df.copy()

# Extract numeric columns only for correlation analysis
numeric_df = eda_df.select_dtypes(include=[np.number])

# If there are at least 4 numeric columns, produce a correlation heatmap
if numeric_df.shape[1] >= 4:
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Numeric Features')
    plt.tight_layout()
    plt.show()
else:
    print('Not enough numeric columns for a meaningful correlation heatmap.')

# Pair Plot for numeric features
sns.pairplot(numeric_df)
plt.suptitle('Pair Plot of Numeric Features', y=1.02)
plt.show()

# Histogram for 'Close' prices
plt.figure(figsize=(8, 6))
sns.histplot(eda_df['Close'], kde=True, color='skyblue')
plt.title('Histogram of Closing Prices')
plt.xlabel('Close Price')
plt.ylabel('Frequency')
plt.show()
# Countplot for categorical variable 'ticker'
plt.figure(figsize=(6, 4))
sns.countplot(x='ticker', data=eda_df, palette='viridis')
plt.title('Count of Records per Ticker')
plt.show()

# Barplot for average volume per ticker
plt.figure(figsize=(6, 4))
sns.barplot(x='ticker', y='Volume', data=eda_df, palette='magma')
plt.title('Average Trading Volume per Ticker')
plt.show()

# Boxplot for 'Close' prices to inspect distribution and outliers
plt.figure(figsize=(8, 6))
sns.boxplot(y='Close', data=eda_df, color='lightgreen')
plt.title('Boxplot of Closing Prices')
plt.show()
# Boxen plot for 'Volume'
plt.figure(figsize=(8, 6))
sns.boxenplot(y='Volume', data=eda_df, palette='pastel')
plt.title('Boxen Plot of Trading Volume')
plt.show()

# Violin plot comparing 'Open' and 'Close' prices
plt.figure(figsize=(8, 6))
sns.violinplot(data=eda_df[['Open', 'Close']], palette='Set2')
plt.title('Violin Plot of Open vs Close Prices')
plt.show()

# Strip plot for 'High' prices
plt.figure(figsize=(8, 6))
sns.stripplot(y='High', data=eda_df, color='orange', jitter=0.25)
plt.title('Strip Plot of High Prices')
plt.show()

# For the prediction task, we'll use a simple Linear Regression model 
# to predict the 'Close' price based on features 'Open', 'High', 'Low', and 'Volume'.

# Define feature columns and target variable
feature_cols = ['Open', 'High', 'Low', 'Volume']
target_col = 'Close'

# Prepare the data (ensure that no missing values exist in these columns)
model_df = df.dropna(subset=feature_cols + [target_col]).copy()

X = model_df[feature_cols]
y = model_df[target_col]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize and train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict on the test set
y_pred = lr_model.predict(X_test)

# Calculate the prediction accuracy using the R-squared metric
r2 = r2_score(y_test, y_pred)
print(f'Linear Regression R-squared score: {r2:.4f}')

# Optional: Create a scatter plot of Actual vs Predicted Close Prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='teal')
plt.xlabel('Actual Close Price')
plt.ylabel('Predicted Close Price')
plt.title('Actual vs Predicted Close Prices')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.tight_layout()
plt.show()
