import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import warnings
import joblib
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*60)
print("USED BIKE PRICES - FEATURE ENGINEERING AND EDA")
print("="*60)

# 1. Load the dataset
print("\n[1] Loading dataset...")
df = pd.read_csv('bikes.csv')
print(f"Dataset Shape: {df.shape}")
print(f"First 5 rows:\n{df.head()}")

# 2. Check dataset information
print("\n[2] Dataset Info:")
print(df.info())

# 3. Check for missing values
print("\n[3] Missing Values:")
print(df.isnull().sum())

# 4. Check for duplicates
print(f"\n[4] Duplicate Rows: {df.duplicated().sum()}")

# 5. Statistical summary
print("\n[5] Statistical Summary:")
print(df.describe())

# 6. Remove duplicates
print("\n[6] Removing duplicates...")
df_clean = df.drop_duplicates()
print(f"Removed {df.shape[0] - df_clean.shape[0]} duplicate rows")
print(f"New shape: {df_clean.shape}")

# 7. Handle missing values
print("\n[7] Handling missing values...")
print("Missing values before handling:")
print(df_clean.isnull().sum())

# Fill missing location with mode
if df_clean['location'].isnull().sum() > 0:
    location_mode = df_clean['location'].mode()[0]
    df_clean['location'].fillna(location_mode, inplace=True)
    print(f"Filled missing location with: {location_mode}")

print("Missing values after initial handling:")
print(df_clean.isnull().sum())

# 8. Feature Engineering - Extract CC from model name
print("\n[8] Extracting CC from model names...")

def extract_cc(model_name):
    models = model_name.split(" ")
    models = " ".join(models[:-1]).lower() if len(models) > 1 else models[0].lower()
    
    # Try to find CC pattern
    cc_match = re.search(r'([0-9]*cc)', models, re.IGNORECASE)
    if cc_match:
        cc_value = cc_match.group(1)
        if cc_value.lower() != 'cc':
            return cc_value.lower()
    
    cc_match = re.search(r'([0-9]*(cc))', models, re.IGNORECASE)
    if cc_match:
        return cc_match.group(1).lower()
    
    # Special cases for specific models
    special_cases = {
        '1000': '1000cc', '310': '310cc', 'apache rtr 200': '200cc',
        'ns200': '200cc', 'rs200': '200cc', '220': '220cc', '400': '400cc',
        '250': '250cc', '125': '125cc', '160': '160cc', '150': '150cc',
        '350': '350cc', '200': '200cc', '100': '100cc', '180': '180cc',
        '110': '110cc', '390': '390cc', '135': '135cc', 'r15': '150cc',
        '650': '650cc', '750': '750cc', '800': '800cc', '300': '300cc',
        '765': '765cc', '883': '883cc', '797': '797cc', '810': '810cc',
        '321': '321cc', '821': '821cc', '120': '120cc', '1745': '1745cc',
        '899': '899cc', '900': '900cc', '302': '302cc', '959': '959cc',
        '600': '600cc', '502': '502cc', 'um renegade': '279cc',
        'hero splendor': '97cc', 'hero passion plus': '97cc',
        'yamaha fz': '150cc', 'honda hornet': '184cc',
        'royal enfield interceptor': '650cc', 'hero passion pro': '113cc',
        'hero passion xpro': '109cc', 'harley-davidson street bob': '1868cc',
        'harley-davidson fat bob': '1868cc', 'harley-davidson fat boy': '1868cc',
        'harley-davidson street rod': '749cc', 'zx-10r': '1000cc',
        'rsv4': '1099cc', 'tvs sport': '109cc', 'tvs star city': '109cc',
        'harley-davidson superlow': '883cc', 'harley-davidson roadster': '1202cc',
        'harley-davidson forty eight': '1202cc', 'harley-davidson night rod special': '1247cc',
        'triumph rocket iii roadster': '2458cc', 'triumph thunderbird lt': '1699cc',
        'kawasaki vulcan s black': '649cc', 'mahindra mojo black pearl': '300cc',
        'ducati diavel carbon': '1198cc', 'triumph tiger explorer': '1215cc',
        'royal enfield continental': '535cc'
    }
    
    for key, value in special_cases.items():
        if key in models:
            return value
    
    return 'unknown'

df_clean['cc'] = df_clean['model_name'].apply(extract_cc)
print("CC extraction completed.")
print(df_clean['cc'].value_counts().head(20))

# 9. Extract Brand from model name
print("\n[9] Extracting brand from model names...")

def extract_brand(model_name):
    brands = ['Royal Enfield', 'Bajaj', 'Hero', 'Honda', 'Yamaha', 'TVS', 
              'Suzuki', 'Kawasaki', 'Hyosung', 'Jawa', 'KTM', 'Ducati', 
              'Triumph', 'Harley-Davidson', 'BMW', 'Benelli', 'Mahindra', 
              'UM', 'Aprilia', 'MV Agusta', 'Indian', 'Victory']
    
    model_lower = model_name.lower()
    for brand in brands:
        if brand.lower() in model_lower:
            return brand
    
    return model_name.split()[0]

df_clean['brand'] = df_clean['model_name'].apply(extract_brand)
print("Brand extraction completed.")
print(df_clean['brand'].value_counts())

# 10. Clean kms_driven column
print("\n[10] Cleaning kms_driven column...")

def clean_kms_driven(kms):
    if pd.isna(kms):
        return np.nan
    
    kms_str = str(kms).lower().strip()
    
    if kms_str in ['mileage', 'yes']:
        return np.nan
    
    numbers = re.findall(r'[0-9,]+', kms_str)
    if numbers:
        return int(numbers[0].replace(',', ''))
    
    return np.nan

df_clean['kms_driven_clean'] = df_clean['kms_driven'].apply(clean_kms_driven)
kms_mean = df_clean['kms_driven_clean'].mean()
df_clean['kms_driven_clean'].fillna(kms_mean, inplace=True)
print(f"Kms driven cleaned. Mean value used for missing: {kms_mean:.2f}")
print(df_clean['kms_driven_clean'].describe())

# 11. Clean mileage column
print("\n[11] Cleaning mileage column...")

def clean_mileage(mileage):
    if pd.isna(mileage):
        return np.nan
    
    mileage_str = str(mileage).lower().strip()
    numbers = re.findall(r'[0-9.]+', mileage_str)
    if numbers:
        try:
            return float(numbers[0])
        except:
            return np.nan
    return np.nan

df_clean['mileage_clean'] = df_clean['mileage'].apply(clean_mileage)
mileage_median = df_clean['mileage_clean'].median()
df_clean['mileage_clean'].fillna(mileage_median, inplace=True)
print(f"Mileage cleaned. Median value used for missing: {mileage_median:.2f}")
print(df_clean['mileage_clean'].describe())

# 12. Clean power column
print("\n[12] Cleaning power column...")

def clean_power(power):
    if pd.isna(power):
        return np.nan
    
    power_str = str(power).lower().strip()
    numbers = re.findall(r'[0-9.]+', power_str)
    if numbers:
        try:
            return float(numbers[0])
        except:
            return np.nan
    return np.nan

df_clean['power_clean'] = df_clean['power'].apply(clean_power)
power_median = df_clean['power_clean'].median()
df_clean['power_clean'].fillna(power_median, inplace=True)
print(f"Power cleaned. Median value used for missing: {power_median:.2f}")
print(df_clean['power_clean'].describe())

# 13. Create additional features
print("\n[13] Creating additional features...")

def cc_to_numeric(cc):
    if cc == 'unknown':
        return np.nan
    numbers = re.findall(r'[0-9]+', cc)
    if numbers:
        return int(numbers[0])
    return np.nan

df_clean['cc_numeric'] = df_clean['cc'].apply(cc_to_numeric)
cc_median = df_clean['cc_numeric'].median()
df_clean['cc_numeric'].fillna(cc_median, inplace=True)

current_year = 2024
df_clean['bike_age'] = current_year - df_clean['model_year']
df_clean['power_to_cc_ratio'] = df_clean['power_clean'] / df_clean['cc_numeric']

print("Additional features created.")
print(df_clean[['bike_age', 'power_to_cc_ratio']].describe())

# 14. Final data type conversions
print("\n[14] Final data type conversions...")

final_columns = {
    'model_name': str,
    'model_year': int,
    'kms_driven_clean': int,
    'owner': str,
    'location': str,
    'mileage_clean': float,
    'power_clean': float,
    'price': int,
    'cc_numeric': int,
    'brand': str,
    'bike_age': int,
    'power_to_cc_ratio': float
}

df_final = df_clean[list(final_columns.keys())].copy()
df_final.columns = ['model_name', 'model_year', 'kms_driven', 'owner', 'location', 
                    'mileage', 'power', 'price', 'cc', 'brand', 'bike_age', 'power_to_cc_ratio']

for col, dtype in final_columns.items():
    try:
        df_final[col] = df_final[col].astype(dtype)
    except:
        pass

print(f"Final dataset shape: {df_final.shape}")
print(df_final.info())

# 15. Descriptive Statistics
print("\n[15] Descriptive Statistics:")
numerical_cols = ['model_year', 'kms_driven', 'mileage', 'power', 'price', 'cc', 'bike_age', 'power_to_cc_ratio']
print(df_final[numerical_cols].describe())

# 16. Create visualizations
print("\n[16] Creating visualizations...")

# Create output directory for plots
import os
os.makedirs('plots', exist_ok=True)

# Price Distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(df_final['price'], kde=True, bins=50)
plt.title('Price Distribution')
plt.xlabel('Price (INR)')
plt.subplot(1, 2, 2)
sns.boxplot(y=df_final['price'])
plt.title('Price Boxplot')
plt.tight_layout()
plt.savefig('plots/price_distribution.png')
plt.close()
print("Saved: plots/price_distribution.png")

# Distribution of numerical features
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
sns.histplot(df_final['kms_driven'], kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Kilometers Driven Distribution')
sns.histplot(df_final['mileage'], kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Mileage Distribution')
sns.histplot(df_final['power'], kde=True, ax=axes[0, 2])
axes[0, 2].set_title('Power Distribution')
sns.histplot(df_final['cc'], kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Engine CC Distribution')
sns.histplot(df_final['bike_age'], kde=True, ax=axes[1, 1])
axes[1, 1].set_title('Bike Age Distribution')
sns.histplot(df_final['power_to_cc_ratio'], kde=True, ax=axes[1, 2])
axes[1, 2].set_title('Power to CC Ratio Distribution')
plt.tight_layout()
plt.savefig('plots/numerical_distributions.png')
plt.close()
print("Saved: plots/numerical_distributions.png")

# Categorical features distribution
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
year_counts = df_final['model_year'].value_counts().sort_index()
axes[0, 0].bar(year_counts.index, year_counts.values)
axes[0, 0].set_title('Year-wise Distribution of Bikes')
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Count')
axes[0, 0].tick_params(axis='x', rotation=45)

brand_counts = df_final['brand'].value_counts().head(15)
axes[0, 1].barh(brand_counts.index, brand_counts.values)
axes[0, 1].set_title('Brand-wise Distribution (Top 15)')
axes[0, 1].set_xlabel('Count')

owner_counts = df_final['owner'].value_counts()
axes[1, 0].bar(owner_counts.index, owner_counts.values)
axes[1, 0].set_title('Owner Type Distribution')
axes[1, 0].set_xlabel('Owner Type')
axes[1, 0].set_ylabel('Count')
axes[1, 0].tick_params(axis='x', rotation=45)

location_counts = df_final['location'].value_counts().head(15)
axes[1, 1].barh(location_counts.index, location_counts.values)
axes[1, 1].set_title('Location Distribution (Top 15)')
axes[1, 1].set_xlabel('Count')
plt.tight_layout()
plt.savefig('plots/categorical_distributions.png')
plt.close()
print("Saved: plots/categorical_distributions.png")

# Bivariate analysis
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_final, x='model_year', y='price', alpha=0.5)
plt.title('Price vs Model Year')
plt.xlabel('Model Year')
plt.ylabel('Price (INR)')
plt.savefig('plots/price_vs_year.png')
plt.close()
print("Saved: plots/price_vs_year.png")

plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_final, x='kms_driven', y='price', alpha=0.5)
plt.title('Price vs Kilometers Driven')
plt.xlabel('Kilometers Driven')
plt.ylabel('Price (INR)')
plt.savefig('plots/price_vs_kms.png')
plt.close()
print("Saved: plots/price_vs_kms.png")

plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_final, x='power', y='price', alpha=0.5)
plt.title('Price vs Power')
plt.xlabel('Power (BHP)')
plt.ylabel('Price (INR)')
plt.savefig('plots/price_vs_power.png')
plt.close()
print("Saved: plots/price_vs_power.png")

plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_final, x='cc', y='price', alpha=0.5)
plt.title('Price vs Engine CC')
plt.xlabel('Engine CC')
plt.ylabel('Price (INR)')
plt.savefig('plots/price_vs_cc.png')
plt.close()
print("Saved: plots/price_vs_cc.png")

# Price by Brand
plt.figure(figsize=(14, 8))
top_brands = df_final['brand'].value_counts().head(10).index
sns.boxplot(data=df_final[df_final['brand'].isin(top_brands)], x='brand', y='price')
plt.title('Price Distribution by Brand (Top 10)')
plt.xlabel('Brand')
plt.ylabel('Price (INR)')
plt.xticks(rotation=45)
plt.savefig('plots/price_by_brand.png')
plt.close()
print("Saved: plots/price_by_brand.png")

# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df_final[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png')
plt.close()
print("Saved: plots/correlation_heatmap.png")

# Average price by location
location_price = df_final.groupby('location')['price'].mean().sort_values(ascending=False).head(15)
plt.figure(figsize=(12, 8))
location_price.plot(kind='barh')
plt.title('Average Bike Price by Location (Top 15)')
plt.xlabel('Average Price (INR)')
plt.ylabel('Location')
plt.savefig('plots/price_by_location.png')
plt.close()
print("Saved: plots/price_by_location.png")

# 17. Feature Engineering for ML
print("\n[17] Preparing features for machine learning...")

features_for_model = df_final[['kms_driven', 'mileage', 'power', 'cc', 'bike_age', 
                               'power_to_cc_ratio', 'brand', 'owner', 'location', 'price']].copy()

le_brand = LabelEncoder()
le_owner = LabelEncoder()
le_location = LabelEncoder()

features_for_model['brand_encoded'] = le_brand.fit_transform(features_for_model['brand'])
features_for_model['owner_encoded'] = le_owner.fit_transform(features_for_model['owner'])
features_for_model['location_encoded'] = le_location.fit_transform(features_for_model['location'])

numerical_features = ['kms_driven', 'mileage', 'power', 'cc', 'bike_age', 'power_to_cc_ratio']
categorical_features = ['brand_encoded', 'owner_encoded', 'location_encoded']

print("Features prepared for modeling.")

# Standardize numerical features
scaler = StandardScaler()
features_scaled = features_for_model.copy()
features_scaled[numerical_features] = scaler.fit_transform(features_for_model[numerical_features])
print("Features standardized.")

# 18. Model Building
print("\n[18] Building machine learning models...")

X = features_scaled[numerical_features + categorical_features]
y = features_scaled['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Linear Regression
print("\n[18.1] Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_pred)
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)
print(f"Linear Regression - MSE: {lr_mse:.4f}, MAE: {lr_mae:.4f}, R²: {lr_r2:.4f}")

# Ridge Regression
print("\n[18.2] Training Ridge Regression...")
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
ridge_pred = ridge_model.predict(X_test)
ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_mae = mean_absolute_error(y_test, ridge_pred)
ridge_r2 = r2_score(y_test, ridge_pred)
print(f"Ridge Regression - MSE: {ridge_mse:.4f}, MAE: {ridge_mae:.4f}, R²: {ridge_r2:.4f}")

# Random Forest
print("\n[18.3] Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)
print(f"Random Forest - MSE: {rf_mse:.4f}, MAE: {rf_mae:.4f}, R²: {rf_r2:.4f}")

# Gradient Boosting
print("\n[18.4] Training Gradient Boosting...")
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_mse = mean_squared_error(y_test, gb_pred)
gb_mae = mean_absolute_error(y_test, gb_pred)
gb_r2 = r2_score(y_test, gb_pred)
print(f"Gradient Boosting - MSE: {gb_mse:.4f}, MAE: {gb_mae:.4f}, R²: {gb_r2:.4f}")

# Model Comparison
print("\n[19] Model Comparison:")
models_comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge Regression', 'Random Forest', 'Gradient Boosting'],
    'MSE': [lr_mse, ridge_mse, rf_mse, gb_mse],
    'MAE': [lr_mae, ridge_mae, rf_mae, gb_mae],
    'R² Score': [lr_r2, ridge_r2, rf_r2, gb_r2]
})
print(models_comparison)

# Plot model comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.barplot(data=models_comparison, x='Model', y='MSE', ax=axes[0])
axes[0].set_title('Model Comparison - MSE')
axes[0].tick_params(axis='x', rotation=45)
sns.barplot(data=models_comparison, x='Model', y='MAE', ax=axes[1])
axes[1].set_title('Model Comparison - MAE')
axes[1].tick_params(axis='x', rotation=45)
sns.barplot(data=models_comparison, x='Model', y='R² Score', ax=axes[2])
axes[2].set_title('Model Comparison - R² Score')
axes[2].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('plots/model_comparison.png')
plt.close()
print("Saved: plots/model_comparison.png")

# Feature Importance
print("\n[20] Feature Importance:")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='Importance', y='Feature')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.savefig('plots/feature_importance.png')
plt.close()
print("Saved: plots/feature_importance.png")

# Hyperparameter Tuning (simplified)
print("\n[21] Hyperparameter Tuning for Random Forest...")
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), 
                          param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best R² Score: {grid_search.best_score_:.4f}")

# Evaluate best model
best_rf_model = grid_search.best_estimator_
best_rf_pred = best_rf_model.predict(X_test)
best_rf_mse = mean_squared_error(y_test, best_rf_pred)
best_rf_mae = mean_absolute_error(y_test, best_rf_pred)
best_rf_r2 = r2_score(y_test, best_rf_pred)
print(f"Best Random Forest - MSE: {best_rf_mse:.4f}, MAE: {best_rf_mae:.4f}, R²: {best_rf_r2:.4f}")

# 22. Save cleaned dataset and model
print("\n[22] Saving cleaned dataset and models...")
df_final.to_csv('bikes_cleaned.csv', index=False)
print("Cleaned dataset saved as 'bikes_cleaned.csv'")

joblib.dump(best_rf_model, 'bike_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_brand, 'label_encoder_brand.pkl')
joblib.dump(le_owner, 'label_encoder_owner.pkl')
joblib.dump(le_location, 'label_encoder_location.pkl')
print("Model and encoders saved successfully.")

print("\n" + "="*60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("="*60)
print("\nGenerated Files:")
print("- bikes_cleaned.csv (cleaned dataset)")
print("- bike_price_model.pkl (trained model)")
print("- scaler.pkl (feature scaler)")
print("- label_encoder_*.pkl (encoders)")
print("- plots/ directory with all visualizations")
print("\nKey Findings:")
print(f"- Best Model: Random Forest with R² Score: {best_rf_r2:.4f}")
print(f"- Most Important Features: {feature_importance.iloc[0]['Feature']}")
print(f"- Dataset Size: {df_final.shape[0]} bikes after cleaning")
print(f"- Number of Brands: {df_final['brand'].nunique()}")
print(f"- Number of Locations: {df_final['location'].nunique()}")
