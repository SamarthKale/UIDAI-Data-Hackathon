import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, r2_score, mean_absolute_error, mean_squared_error
import warnings
import glob
import os
from datetime import datetime
import json
import time

warnings.filterwarnings('ignore')

# =============================================================================
# LOGGING & PROGRESS HELPERS
# =============================================================================

def log_step(step_name):
    print(f"\n⏳ {step_name}...")
    return time.time()

def end_step(start_time):
    elapsed = time.time() - start_time
    print(f"✅ Completed in {elapsed:.2f} seconds")

def log_plot(plot_name):
    print(f"⏳ Generating plot: {plot_name}...")
    return time.time()

def end_plot(start_time, plot_name):
    elapsed = time.time() - start_time
    print(f"✅ Saved {plot_name} ({elapsed:.2f}s)")

# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================

os.makedirs('cleaned_data', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# =============================================================================
# PIPELINE HEADER
# =============================================================================

print("="*100)
print(" "*30 + "🏆 AADHAAR DEMOGRAPHIC DATA ANALYSIS 🏆")
print("="*100)

# ============================================================================
# 1. DATA LOADING & CLEANING
# ============================================================================

print("\n[1] DATA LOADING & CLEANING")
print("-"*100)

# Load data
file_pattern = 'data/api_data_aadhar_demographic/api_data_aadhar_demographic/*.csv'
csv_files = glob.glob(file_pattern)

if not csv_files:
    print(f"⚠️  No files found matching pattern: {file_pattern}")
    exit()

print(f"✓ Found {len(csv_files)} CSV files")

# Load and merge files
dfs = []
for file in csv_files:
    try:
        df_temp = pd.read_csv(file, low_memory=False)
        dfs.append(df_temp)
        print(f"  ✓ Loaded {os.path.basename(file)}: {len(df_temp):,} records")
    except Exception as e:
        print(f"  ✗ Error loading {file}: {e}")

raw_df = pd.concat(dfs, ignore_index=True)
print(f"\n✓ Total raw records: {len(raw_df):,}")

# Save raw data summary
raw_df.to_csv('cleaned_data/00_raw_data.csv', index=False)
print("✓ Saved raw data as 'cleaned_data/00_raw_data.csv'")

# ============================================================================
# DATA QUALITY REPORT
# ============================================================================

print("\n--- DATA QUALITY REPORT ---")

def generate_quality_report(df, name):
    report = {
        'dataset': name,
        'total_records': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        'duplicate_rows': df.duplicated().sum(),
        'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100,
        'column_details': {}
    }
    
    for col in df.columns:
        report['column_details'][col] = {
            'dtype': str(df[col].dtype),
            'unique_values': df[col].nunique(),
            'null_count': df[col].isnull().sum(),
            'null_percentage': (df[col].isnull().sum() / len(df)) * 100
        }
    
    return report

quality_report = generate_quality_report(raw_df, "Raw Data")
print(f"Data Quality Score: {100 - quality_report['missing_percentage']:.2f}%")
print(f"Duplicate Records: {quality_report['duplicate_rows']:,} ({quality_report['duplicate_percentage']:.2f}%)")

# ============================================================================
# DATA CLEANING PIPELINE
# ============================================================================

print("\n--- DATA CLEANING PIPELINE ---")

# Step 1: Remove duplicates
initial_count = len(raw_df)
df_clean = raw_df.drop_duplicates()
print(f"✓ Removed {initial_count - len(df_clean):,} duplicate records")

# Step 2: Parse dates
date_col = 'date'  # Adjust based on your actual column name
if date_col in df_clean.columns:
    # Try multiple date formats
    for fmt in ['%d-%m-%Y', '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y']:
        try:
            df_clean['date'] = pd.to_datetime(df_clean['date'], format=fmt, errors='coerce')
            if df_clean['date'].notna().any():
                print(f"✓ Parsed dates with format: {fmt}")
                break
        except:
            continue
    
    # If still have nulls, try inferring
    if df_clean['date'].isnull().any():
        df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
    
    invalid_dates = df_clean['date'].isnull().sum()
    if invalid_dates > 0:
        print(f"⚠️  Found {invalid_dates} invalid dates")
        df_clean = df_clean.dropna(subset=['date'])
else:
    print("⚠️  No date column found")
    # Create a dummy date column if needed for analysis
    df_clean['date'] = pd.Timestamp('2023-01-01')

# Step 3: Clean text columns
text_cols = ['state', 'district']
for col in text_cols:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].astype(str).str.strip().str.title()
        df_clean[col] = df_clean[col].str.replace(r'\s+', ' ', regex=True)

# Step 4: Handle numeric columns
# Identify numeric columns dynamically
numeric_cols = []
for col in df_clean.columns:
    if df_clean[col].dtype in ['int64', 'float64']:
        numeric_cols.append(col)
    elif col in ['demo_age_5_17', 'demo_age_17_']:  # Known numeric columns
        numeric_cols.append(col)

# Convert known numeric columns
for col in ['demo_age_5_17', 'demo_age_17_']:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        null_count = df_clean[col].isnull().sum()
        if null_count > 0:
            print(f"  {col}: Imputing {null_count:,} missing values with median")
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# Step 5: Outlier detection
print("\n--- Outlier Detection ---")
if len(numeric_cols) >= 2:
    try:
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        outlier_features = df_clean[numeric_cols[:2]].values  # Use first 2 numeric columns
        outlier_pred = iso_forest.fit_predict(outlier_features)
        outlier_count = (outlier_pred == -1).sum()
        print(f"  Detected {outlier_count:,} outliers ({outlier_count/len(df_clean)*100:.2f}%)")
        
        # Save outliers separately
        df_clean['is_outlier'] = outlier_pred == -1
        outliers_df = df_clean[df_clean['is_outlier']].copy()
        df_clean = df_clean[~df_clean['is_outlier']]
        df_clean = df_clean.drop('is_outlier', axis=1)
        
        outliers_df.to_csv('cleaned_data/01_outliers_removed.csv', index=False)
        print("✓ Saved outliers data")
    except Exception as e:
        print(f"  ⚠️  Outlier detection failed: {e}")

# Save cleaned data
df_clean.to_csv('cleaned_data/02_cleaned_data.csv', index=False)
print("\n✓ Saved cleaned data as 'cleaned_data/02_cleaned_data.csv'")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

print("\n--- Feature Engineering ---")

# Check for required columns before feature engineering
required_cols = ['demo_age_5_17', 'demo_age_17_', 'date']
missing_cols = [col for col in required_cols if col not in df_clean.columns]

if missing_cols:
    print(f"⚠️  Missing columns for feature engineering: {missing_cols}")
    print("  Skipping advanced feature engineering")
else:
    # Basic features
    df_clean['total_enrollments'] = df_clean['demo_age_5_17'] + df_clean['demo_age_17_']
    df_clean['child_ratio'] = df_clean['demo_age_5_17'] / (df_clean['total_enrollments'] + 1e-10)  # Avoid division by zero
    df_clean['adult_ratio'] = df_clean['demo_age_17_'] / (df_clean['total_enrollments'] + 1e-10)
    
    # Temporal features
    df_clean['year'] = df_clean['date'].dt.year
    df_clean['month'] = df_clean['date'].dt.month
    df_clean['quarter'] = df_clean['date'].dt.quarter
    df_clean['day_of_week'] = df_clean['date'].dt.dayofweek
    df_clean['week_of_year'] = df_clean['date'].dt.isocalendar().week
    df_clean['is_weekend'] = df_clean['day_of_week'].isin([5, 6]).astype(int)
    
    # Transformation features
    df_clean['log_enrollments'] = np.log1p(df_clean['total_enrollments'])
    
    print(f"✓ Created {len([col for col in df_clean.columns if col not in raw_df.columns])} new features")
    
    # Save feature-engineered data
    df_clean.to_csv('cleaned_data/03_feature_engineered.csv', index=False)
    print("✓ Saved feature-engineered data")

# ============================================================================
# 2. CURRENT ANALYSIS (ACTUAL DATA)
# ============================================================================

print("\n\n[2] CURRENT ANALYSIS - ACTUAL DATA")
print("-"*100)

# ANALYSIS 1: Temporal Trends
print("\n--- ANALYSIS 1: Temporal Enrollment Trends ---")

if 'date' in df_clean.columns and 'total_enrollments' in df_clean.columns:
    try:
        # Daily trends
        daily_enrollments = df_clean.groupby('date')['total_enrollments'].sum()
        
        # Check if we have any data
        if len(daily_enrollments) > 0:
            # Visualization 1: Daily Trends
            plt.figure(figsize=(15, 6))
            plt.plot(daily_enrollments.index, daily_enrollments.values, alpha=0.7, linewidth=1)
            plt.title('Daily Enrollment Trends', fontsize=16, fontweight='bold')
            plt.xlabel('Date')
            plt.ylabel('Total Enrollments')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('visualizations/01_daily_enrollment_trends.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved: Daily Enrollment Trends")
            
            # Monthly trends
            monthly_enrollments = df_clean.groupby(df_clean['date'].dt.to_period('M'))['total_enrollments'].sum()
            monthly_enrollments.index = monthly_enrollments.index.to_timestamp()
            
            # Visualization 2: Monthly Trends
            plt.figure(figsize=(15, 6))
            plt.bar(monthly_enrollments.index, monthly_enrollments.values, alpha=0.7)
            plt.title('Monthly Enrollment Trends', fontsize=16, fontweight='bold')
            plt.xlabel('Month')
            plt.ylabel('Total Enrollments')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig('visualizations/02_monthly_enrollment_trends.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved: Monthly Enrollment Trends")
            
            # Statistics
            print(f"\n  Monthly Statistics:")
            print(f"    Average Monthly Enrollments: {monthly_enrollments.mean():,.0f}")
            print(f"    Best Month: {monthly_enrollments.idxmax().strftime('%B %Y')} - {monthly_enrollments.max():,.0f}")
            print(f"    Worst Month: {monthly_enrollments.idxmin().strftime('%B %Y')} - {monthly_enrollments.min():,.0f}")
        else:
            print("⚠️  No temporal data available for analysis")
    except Exception as e:
        print(f"⚠️  Error in temporal analysis: {e}")

# ANALYSIS 2: Geographic Analysis
print("\n--- ANALYSIS 2: Geographic Distribution ---")

if 'state' in df_clean.columns and 'total_enrollments' in df_clean.columns:
    try:
        state_analysis = df_clean.groupby('state').agg({
            'total_enrollments': ['sum', 'mean', 'count'],
            'child_ratio': 'mean' if 'child_ratio' in df_clean.columns else None
        }).round(2)
        
        # Handle the case where child_ratio might not exist
        if 'child_ratio' in df_clean.columns:
            state_analysis.columns = ['total_enrollments', 'avg_enrollments', 'record_count', 'avg_child_ratio']
        else:
            state_analysis.columns = ['total_enrollments', 'avg_enrollments', 'record_count']
        
        state_analysis = state_analysis.sort_values('total_enrollments', ascending=False)
        
        # Save state analysis
        state_analysis.to_csv('cleaned_data/04_state_analysis.csv')
        print("✓ Saved: State Analysis CSV")
        
        # Visualization 3: Top States
        plt.figure(figsize=(12, 8))
        top_states = state_analysis.head(min(15, len(state_analysis)))
        colors = plt.cm.Set3(np.arange(len(top_states)) / len(top_states))
        bars = plt.barh(range(len(top_states)), top_states['total_enrollments'].values, color=colors)
        plt.yticks(range(len(top_states)), top_states.index)
        plt.xlabel('Total Enrollments')
        plt.title('Top States by Total Enrollments', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig('visualizations/03_top_states.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: Top States Visualization")
        
        # Visualization 4: Child Ratio by State (if available)
        if 'avg_child_ratio' in state_analysis.columns:
            plt.figure(figsize=(12, 8))
            child_ratio_sorted = state_analysis.sort_values('avg_child_ratio', ascending=False).head(min(15, len(state_analysis)))
            plt.bar(range(len(child_ratio_sorted)), child_ratio_sorted['avg_child_ratio'].values * 100)
            plt.xticks(range(len(child_ratio_sorted)), child_ratio_sorted.index, rotation=45, ha='right')
            plt.ylabel('Child Ratio (%)')
            plt.title('States by Child Enrollment Ratio', fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig('visualizations/04_child_ratio_by_state.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved: Child Ratio by State")
    except Exception as e:
        print(f"⚠️  Error in geographic analysis: {e}")

# ANALYSIS 3: Age Group Analysis
print("\n--- ANALYSIS 3: Age Group Distribution ---")

if all(col in df_clean.columns for col in ['demo_age_5_17', 'demo_age_17_']):
    try:
        total_child = df_clean['demo_age_5_17'].sum()
        total_adult = df_clean['demo_age_17_'].sum()
        
        if total_child + total_adult > 0:
            # Visualization 5: Age Group Pie Chart
            plt.figure(figsize=(10, 8))
            labels = ['Age 5-17', 'Age 17+']
            sizes = [total_child, total_adult]
            colors = ['#ff9999', '#66b3ff']
            explode = (0.1, 0)
            
            plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                    autopct='%1.1f%%', shadow=True, startangle=90)
            plt.axis('equal')
            plt.title('Overall Age Group Distribution', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig('visualizations/05_age_group_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved: Age Group Distribution")
            
            print(f"\n  Age Group Statistics:")
            print(f"    Total Child Enrollments (5-17): {total_child:,.0f}")
            print(f"    Total Adult Enrollments (17+): {total_adult:,.0f}")
            print(f"    Child Percentage: {(total_child/(total_child+total_adult)*100):.1f}%")
        else:
            print("⚠️  No age group data available")
    except Exception as e:
        print(f"⚠️  Error in age group analysis: {e}")

# ANALYSIS 4: Monthly Patterns (FIXED VERSION)
print("\n--- ANALYSIS 4: Monthly Patterns ---")

if 'month' in df_clean.columns and 'total_enrollments' in df_clean.columns:
    try:
        # Get available months and their data
        monthly_pattern = df_clean.groupby('month')['total_enrollments'].mean().sort_index()
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        print(f"  Data available for {len(monthly_pattern)} months: {list(monthly_pattern.index)}")
        
        if len(monthly_pattern) > 0:
            # Visualization 6: Monthly Pattern
            plt.figure(figsize=(12, 6))
            
            # Use actual month numbers from data
            available_months = monthly_pattern.index.tolist()
            month_labels = [month_names[m-1] for m in available_months]
            
            plt.plot(available_months, monthly_pattern.values, marker='o', linewidth=2, markersize=8)
            plt.xticks(available_months, month_labels)
            plt.xlabel('Month')
            plt.ylabel('Average Enrollments')
            plt.title('Average Enrollment Pattern by Month', fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            # Add data labels
            for i, (month, value) in enumerate(zip(available_months, monthly_pattern.values)):
                plt.text(month, value, f'{value:,.0f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig('visualizations/06_monthly_pattern.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved: Monthly Pattern")
            
            # Print monthly statistics
            print(f"\n  Monthly Statistics:")
            for month, value in monthly_pattern.items():
                month_name = month_names[month-1]
                print(f"    {month_name}: {value:,.0f} enrollments")
        else:
            print("⚠️  No monthly pattern data available")
    except Exception as e:
        print(f"⚠️  Error in monthly pattern analysis: {e}")
        import traceback
        traceback.print_exc()

# ANALYSIS 5: Weekly Patterns
print("\n--- ANALYSIS 5: Weekly Patterns ---")

if 'day_of_week' in df_clean.columns and 'total_enrollments' in df_clean.columns:
    try:
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_pattern = df_clean.groupby('day_of_week')['total_enrollments'].mean()
        
        if len(weekly_pattern) > 0:
            # Visualization 7: Weekly Pattern
            plt.figure(figsize=(10, 6))
            bars = plt.bar(range(len(weekly_pattern)), weekly_pattern.values, color='skyblue', edgecolor='black')
            plt.xticks(range(len(weekly_pattern)), [d[:3] for d in day_names[:len(weekly_pattern)]])
            plt.xlabel('Day of Week')
            plt.ylabel('Average Enrollments')
            plt.title('Average Enrollment Pattern by Day of Week', fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:,.0f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig('visualizations/07_weekly_pattern.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved: Weekly Pattern")
        else:
            print("⚠️  No weekly pattern data available")
    except Exception as e:
        print(f"⚠️  Error in weekly pattern analysis: {e}")

# ============================================================================
# 3. ADVANCED ANALYSIS (CLUSTERING & SEGMENTATION) — WITH PROGRESS
# ============================================================================

print("\n\n[3] ADVANCED ANALYSIS - CLUSTERING (WITH PROGRESS)")
print("-"*100)

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

step_time = log_step("Preparing data for clustering")

required_cluster_cols = ['total_enrollments', 'child_ratio', 'demo_age_5_17', 'demo_age_17_']
available_cluster_cols = [col for col in required_cluster_cols if col in df_clean.columns]

if len(available_cluster_cols) >= 2:
    try:
        X_cluster = df_clean[available_cluster_cols].dropna()
        print(f"  • Records for clustering: {len(X_cluster):,}")
        print(f"  • Features used: {available_cluster_cols}")

        if len(X_cluster) >= 100:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_cluster)

            # Optional PCA
            if X_scaled.shape[1] > 3:
                print("  • Applying PCA compression...")
                pca = PCA(n_components=3)
                X_scaled = pca.fit_transform(X_scaled)

            end_step(step_time)

            # Find optimal clusters
            step_time = log_step("Finding optimal number of clusters")

            max_clusters = min(7, len(X_scaled)//500)
            max_clusters = max(3, max_clusters)

            silhouette_scores = []
            K_range = range(2, max_clusters)

            for idx, k in enumerate(K_range, 1):
                print(f"  ▶ Running KMeans for K={k} ({idx}/{len(K_range)})")

                model = MiniBatchKMeans(n_clusters=k, batch_size=1024, random_state=42)
                labels = model.fit_predict(X_scaled)

                sil = silhouette_score(
                    X_scaled,
                    labels,
                    sample_size=min(5000, len(X_scaled)),
                    random_state=42
                )

                silhouette_scores.append(sil)
                print(f"     Silhouette Score: {sil:.4f}")

            optimal_k = K_range[silhouette_scores.index(max(silhouette_scores))]
            print(f"\n🏆 Optimal clusters found: {optimal_k}")

            end_step(step_time)

            # Final clustering
            step_time = log_step("Running final clustering model")

            final_model = MiniBatchKMeans(n_clusters=optimal_k, batch_size=1024, random_state=42)
            final_labels = final_model.fit_predict(X_scaled)

            df_clustered = df_clean.loc[X_cluster.index].copy()
            df_clustered['cluster'] = final_labels

            df_clustered.to_csv('cleaned_data/05_clustered_data.csv', index=False)
            print("  ✓ Clustered data saved")

            end_step(step_time)

            # Cluster profiles
            step_time = log_step("Generating cluster profiles")

            for cid in sorted(df_clustered['cluster'].unique()):
                cdata = df_clustered[df_clustered['cluster'] == cid]
                print(f"\n  Cluster {cid}")
                print(f"    Size: {len(cdata):,}")
                print(f"    Avg Enrollments: {cdata['total_enrollments'].mean():,.0f}")
                if 'child_ratio' in cdata.columns:
                    print(f"    Avg Child Ratio: {cdata['child_ratio'].mean():.3f}")

            end_step(step_time)

        else:
            print("⚠️  Not enough data for clustering (need at least 100 rows)")

    except Exception as e:
        print(f"⚠️  Error in clustering analysis: {e}")
        import traceback
        traceback.print_exc()
else:
    print("⚠️  Not enough features available for clustering")

# ============================================================================
# 4. FUTURE ANALYSIS (PREDICTIVE MODELING) — WITH PROGRESS
# ============================================================================

print("\n\n[4] FUTURE ANALYSIS - PREDICTIVE MODELING")
print("-"*100)

import time
from sklearn.preprocessing import LabelEncoder

start_time = time.time()

prediction_features = ['year', 'month', 'quarter', 'day_of_week', 'state', 'total_enrollments']
available_pred_features = [col for col in prediction_features if col in df_clean.columns]

if 'total_enrollments' in available_pred_features and len(available_pred_features) >= 3:
    try:
        print("\n--- Enrollment Prediction Model ---")

        df_pred = df_clean.copy()

        # Encode state safely
        if 'state' in df_pred.columns:
            print("⏳ Encoding state column...")
            le = LabelEncoder()
            df_pred['state_code'] = le.fit_transform(df_pred['state'].astype(str))

        # Define features
        features = ['year', 'month', 'quarter', 'day_of_week', 'state_code'] if 'state_code' in df_pred.columns else ['year', 'month', 'quarter', 'day_of_week']
        features = [f for f in features if f in df_pred.columns]
        target = 'total_enrollments'

        X = df_pred[features]
        y = df_pred[target]

        # Remove nulls
        valid_idx = X.notna().all(axis=1) & y.notna()
        X = X[valid_idx]
        y = y[valid_idx]

        if len(X) >= 20:
            print(f"✓ Total samples: {len(X):,}")
            print(f"✓ Features used: {features}")

            # Temporal split
            split_idx = int(len(X) * 0.8)

            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            print(f"⏳ Training samples: {len(X_train):,}")
            print(f"⏳ Testing samples: {len(X_test):,}")

            print("\n⏳ Training Random Forest model...")

            rf_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=50,
                random_state=42,
                n_jobs=-1
            )

            rf_model.fit(X_train, y_train)

            print("✓ Model training completed")

            # Predictions
            y_train_pred = rf_model.predict(X_train)
            y_test_pred = rf_model.predict(X_test)

            # Metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            print("\n📊 Model Performance:")
            print(f"   Train R² : {train_r2:.4f}")
            print(f"   Test  R² : {test_r2:.4f}")
            print(f"   Test MAE: {test_mae:,.0f}")
            print(f"   Test RMSE: {test_rmse:,.0f}")

            # Feature importance
            fi = pd.DataFrame({
                'Feature': features,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=False)

            print("\n🔍 Feature Importance:")
            for _, row in fi.iterrows():
                print(f"   {row['Feature']}: {row['Importance']:.4f}")

            # Save predictions
            predictions_df = pd.DataFrame({
                'actual': y_test.values,
                'predicted': y_test_pred,
                'error': y_test.values - y_test_pred,
            })

            predictions_df.to_csv('cleaned_data/06_predictions.csv', index=False)
            print("✓ Predictions saved")

            print(f"\n⏱ Prediction Pipeline Time: {(time.time() - start_time):.2f} seconds")

        else:
            print("⚠️ Not enough samples for prediction modeling (need at least 20)")

    except Exception as e:
        print(f"⚠️ Error in prediction modeling: {e}")
        import traceback
        traceback.print_exc()
else:
    print("⚠️ Not enough features available for prediction modeling")

# ============================================================================
# EXTRA VISUAL ANALYTICS (WITH LIVE LOGGING & CONFIRMATION) — FIXED
# ============================================================================

print("\n[5] EXTRA VISUAL ANALYTICS")
print("-"*100)

try:
    # Merge prediction results with dates for time plots
    pred_viz_df = df_pred.loc[y_test.index].copy()
    pred_viz_df['actual'] = y_test.values
    pred_viz_df['predicted'] = y_test_pred
    pred_viz_df['residual'] = pred_viz_df['actual'] - pred_viz_df['predicted']

    # -------------------------------------------------
    # Plot 11: Prediction Error Distribution
    # -------------------------------------------------
    plot_name = "11_prediction_error_distribution.png"
    t = log_plot("Prediction Error Distribution")

    plt.figure(figsize=(10,6))
    plt.hist(pred_viz_df['residual'], bins=50, alpha=0.7)
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("Prediction Error Distribution", fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"visualizations/{plot_name}", dpi=300)
    plt.close()

    end_plot(t, plot_name)
    print(f"📁 File saved: visualizations/{plot_name}")

    # -------------------------------------------------
    # Plot 12: Residual Trend Over Time
    # -------------------------------------------------
    if 'date' in pred_viz_df.columns:
        plot_name = "12_residuals_over_time.png"
        t = log_plot("Residual Trend Over Time")

        plt.figure(figsize=(14,6))
        plt.plot(pred_viz_df['date'], pred_viz_df['residual'], alpha=0.6)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("Date")
        plt.ylabel("Residual Error")
        plt.title("Prediction Residuals Over Time", fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"visualizations/{plot_name}", dpi=300)
        plt.close()

        end_plot(t, plot_name)
        print(f"📁 File saved: visualizations/{plot_name}")

    # -------------------------------------------------
    # Plot 13: Actual vs Predicted Density
    # -------------------------------------------------
    plot_name = "13_actual_vs_predicted_density.png"
    t = log_plot("Actual vs Predicted Density")

    plt.figure(figsize=(8,8))
    plt.scatter(pred_viz_df['actual'], pred_viz_df['predicted'], alpha=0.4, s=10)
    plt.plot(
        [pred_viz_df['actual'].min(), pred_viz_df['actual'].max()],
        [pred_viz_df['actual'].min(), pred_viz_df['actual'].max()],
        'r--'
    )
    plt.xlabel("Actual Enrollments")
    plt.ylabel("Predicted Enrollments")
    plt.title("Actual vs Predicted Distribution", fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"visualizations/{plot_name}", dpi=300)
    plt.close()

    end_plot(t, plot_name)
    print(f"📁 File saved: visualizations/{plot_name}")

    print("\n📊 Extra visual analytics completed successfully!")

except Exception as e:
    print(f"⚠️ Error in extra visualization section: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 5. ADVANCED VISUAL ANALYTICS & ML DIAGNOSTICS (FINAL)
# ============================================================================

print("\n[5] ADVANCED VISUAL ANALYTICS & ML DIAGNOSTICS")
print("-"*100)

try:
    # =================================================
    # DATA VISUAL ANALYTICS
    # =================================================

    # Plot 11: Enrollment Distribution
    t = log_plot("Enrollment Distribution Histogram")
    plt.figure(figsize=(10,6))
    plt.hist(df_clean['total_enrollments'], bins=50, alpha=0.7)
    plt.xlabel("Total Enrollments")
    plt.ylabel("Frequency")
    plt.title("Enrollment Distribution", fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("visualizations/11_enrollment_distribution.png", dpi=300)
    plt.close()
    end_plot(t, "11_enrollment_distribution.png")

    # Plot 12: State-wise Enrollment Boxplot
    if 'state' in df_clean.columns:
        t = log_plot("State-wise Enrollment Boxplot")
        top_states = df_clean.groupby('state')['total_enrollments'].sum().sort_values(ascending=False).head(10).index
        state_df = df_clean[df_clean['state'].isin(top_states)]

        plt.figure(figsize=(14,6))
        sns.boxplot(x='state', y='total_enrollments', data=state_df)
        plt.xticks(rotation=45)
        plt.title("State-wise Enrollment Distribution (Top 10 States)", fontweight='bold')
        plt.tight_layout()
        plt.savefig("visualizations/12_state_enrollment_boxplot.png", dpi=300)
        plt.close()
        end_plot(t, "12_state_enrollment_boxplot.png")

    # Plot 13: Daily Enrollment with Rolling Average
    if 'date' in df_clean.columns:
        t = log_plot("Daily Enrollment Rolling Trend")
        daily = df_clean.groupby('date')['total_enrollments'].sum().sort_index()
        rolling = daily.rolling(7).mean()

        plt.figure(figsize=(14,6))
        plt.plot(daily.index, daily.values, alpha=0.4, label="Daily")
        plt.plot(rolling.index, rolling.values, linewidth=2, label="7-Day Avg")
        plt.legend()
        plt.title("Daily Enrollment with Rolling Average", fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("visualizations/13_enrollment_rolling_trend.png", dpi=300)
        plt.close()
        end_plot(t, "13_enrollment_rolling_trend.png")

    # Plot 14: Feature Correlation Heatmap
    t = log_plot("Feature Correlation Heatmap")
    corr_cols = ['total_enrollments', 'demo_age_5_17', 'demo_age_17_', 'child_ratio']
    corr_cols = [c for c in corr_cols if c in df_clean.columns]
    corr = df_clean[corr_cols].corr()

    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap", fontweight='bold')
    plt.tight_layout()
    plt.savefig("visualizations/14_feature_correlation_heatmap.png", dpi=300)
    plt.close()
    end_plot(t, "14_feature_correlation_heatmap.png")

    # =================================================
    # ML PREDICTION VISUAL DIAGNOSTICS
    # =================================================

    if 'y_test_pred' in locals():

        pred_df = df_pred.iloc[y_test.index].copy()
        pred_df['actual'] = y_test.values
        pred_df['predicted'] = y_test_pred
        pred_df['residual'] = pred_df['actual'] - pred_df['predicted']
        pred_df['error_pct'] = (pred_df['residual'] / (pred_df['actual'] + 1)) * 100

        # Plot 15: Actual vs Predicted
        t = log_plot("Actual vs Predicted Scatter")
        plt.figure(figsize=(8,8))
        plt.scatter(pred_df['actual'], pred_df['predicted'], alpha=0.4)
        plt.plot([pred_df['actual'].min(), pred_df['actual'].max()],
                 [pred_df['actual'].min(), pred_df['actual'].max()], 'r--')
        plt.xlabel("Actual Enrollments")
        plt.ylabel("Predicted Enrollments")
        plt.title("Actual vs Predicted", fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("visualizations/15_actual_vs_predicted.png", dpi=300)
        plt.close()
        end_plot(t, "15_actual_vs_predicted.png")

        # Plot 16: Residual Distribution
        t = log_plot("Residual Distribution")
        plt.figure(figsize=(10,6))
        plt.hist(pred_df['residual'], bins=50, alpha=0.7)
        plt.xlabel("Residual Error")
        plt.ylabel("Frequency")
        plt.title("Prediction Residual Distribution", fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("visualizations/16_residual_distribution.png", dpi=300)
        plt.close()
        end_plot(t, "16_residual_distribution.png")

        # Plot 17: Residuals Over Time
        if 'date' in pred_df.columns:
            t = log_plot("Residuals Over Time")
            plt.figure(figsize=(14,6))
            plt.plot(pred_df['date'], pred_df['residual'], alpha=0.6)
            plt.axhline(0, color='red', linestyle='--')
            plt.title("Residual Trend Over Time", fontweight='bold')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig("visualizations/17_residuals_over_time.png", dpi=300)
            plt.close()
            end_plot(t, "17_residuals_over_time.png")

        # Plot 18: Prediction Error Percentage
        t = log_plot("Prediction Error Percentage")
        plt.figure(figsize=(10,6))
        plt.hist(pred_df['error_pct'], bins=50, alpha=0.7)
        plt.xlabel("Error Percentage")
        plt.ylabel("Frequency")
        plt.title("Prediction Error Percentage Distribution", fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("visualizations/18_error_percentage.png", dpi=300)
        plt.close()
        end_plot(t, "18_error_percentage.png")

    print("\n📊 All advanced visual analytics saved successfully!")

except Exception as e:
    print(f"⚠️ Error in visualization pipeline: {e}")
# ============================================================================
# 6. COMPREHENSIVE REPORT GENERATION
# ============================================================================

print("\n\n[6] GENERATING COMPREHENSIVE REPORT")
print("-"*100)

try:
    # Generate executive summary
    executive_summary = {
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_summary": {
            "total_records_analyzed": len(df_clean),
            "analysis_period": f"{df_clean['date'].min().strftime('%Y-%m-%d')} to {df_clean['date'].max().strftime('%Y-%m-%d')}" if 'date' in df_clean.columns else "Unknown",
            "total_states": df_clean['state'].nunique() if 'state' in df_clean.columns else 0,
            "total_enrollments": df_clean['total_enrollments'].sum() if 'total_enrollments' in df_clean.columns else 0,
            "child_enrollment_percentage": (df_clean['demo_age_5_17'].sum() / df_clean['total_enrollments'].sum() * 100) 
            if all(col in df_clean.columns for col in ['demo_age_5_17', 'total_enrollments']) and df_clean['total_enrollments'].sum() > 0 else 0
        },
        "key_insights": {
            "best_performing_state": state_analysis.index[0] if 'state_analysis' in locals() and len(state_analysis) > 0 else "N/A",
            "average_daily_enrollments": daily_enrollments.mean() if 'daily_enrollments' in locals() and len(daily_enrollments) > 0 else 0,
            "prediction_accuracy": test_r2 if 'test_r2' in locals() else "N/A"
        },
        "files_generated": {
            "cleaned_data_files": len([f for f in os.listdir('cleaned_data') if f.endswith('.csv')]),
            "visualizations": len([f for f in os.listdir('visualizations') if f.endswith('.png')]),
            "reports": len([f for f in os.listdir('reports')])
        }
    }

    # Save executive summary
    with open('reports/executive_summary.json', 'w') as f:
        json.dump(executive_summary, f, indent=2, default=str)

    print("✓ Generated executive summary report")

    # Create README file
    readme_content = f"""
AADHAAR DEMOGRAPHIC DATA ANALYSIS
==================================

Analysis Date: {executive_summary['analysis_date']}

DATA SUMMARY
------------
• Total Records Analyzed: {executive_summary['data_summary']['total_records_analyzed']:,}
• Analysis Period: {executive_summary['data_summary']['analysis_period']}
• Total States: {executive_summary['data_summary']['total_states']}
• Total Enrollments: {executive_summary['data_summary']['total_enrollments']:,.0f}
• Child Enrollment %: {executive_summary['data_summary']['child_enrollment_percentage']:.1f}%

ANALYSIS COMPLETED
------------------
• Data Cleaning: ✓
• Feature Engineering: ✓
• Temporal Analysis: ✓
• Geographic Analysis: ✓
• Age Group Analysis: ✓
• Monthly Pattern Analysis: ✓
• Weekly Pattern Analysis: ✓
• Customer Segmentation: {'✓' if 'cluster' in df_clean.columns else '✗'}
• Predictive Modeling: {'✓' if 'test_r2' in locals() else '✗'}

OUTPUT FILES
------------
Cleaned Data Files: {executive_summary['files_generated']['cleaned_data_files']}
Visualizations: {executive_summary['files_generated']['visualizations']}
Reports: {executive_summary['files_generated']['reports']}

HOW TO USE
----------
1. Check 'cleaned_data/' folder for processed CSV files
2. View 'visualizations/' folder for analysis charts
3. Read 'reports/executive_summary.json' for key insights

NOTE: Some analyses may be limited due to data availability
"""

    with open('reports/README.md', 'w') as f:
        f.write(readme_content)

    print("✓ Generated README file")

except Exception as e:
    print(f"⚠️  Error generating reports: {e}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*100)
print("ANALYSIS COMPLETE - SUMMARY")
print("="*100)

# Count actual files generated
cleaned_files = len([f for f in os.listdir('cleaned_data') if f.endswith('.csv')])
visualization_files = len([f for f in os.listdir('visualizations') if f.endswith('.png')])

print(f"""
📁 OUTPUT FOLDERS:
  • cleaned_data/ - Contains {cleaned_files} cleaned CSV files
  • visualizations/ - Contains {visualization_files} individual graphs
  • reports/ - Contains analysis reports

📊 DATA PROCESSED:
  • Raw Records: {len(raw_df):,}
  • Cleaned Records: {len(df_clean):,}
  • Data Quality: {(df_clean.isnull().sum().sum()/(len(df_clean)*max(1, len(df_clean.columns))))*100:.1f}% missing values

📈 ANALYSIS COMPLETED:
  1. Data Loading & Cleaning ✓
  2. Feature Engineering ✓
  3. Temporal Analysis ✓
  4. Geographic Analysis ✓
  5. Demographic Analysis ✓
  6. Pattern Analysis ✓
  7. Advanced Clustering {'✓' if 'cluster' in df_clean.columns else '✗'}
  8. Predictive Modeling {'✓' if 'test_r2' in locals() else '✗'}

🎯 NEXT STEPS:
  1. Review visualizations in 'visualizations/' folder
  2. Check cleaned data in 'cleaned_data/' folder
  3. Read reports in 'reports/' folder
""")

print("="*100)
print("✅ ANALYSIS PIPELINE COMPLETED!")
print("="*100)

