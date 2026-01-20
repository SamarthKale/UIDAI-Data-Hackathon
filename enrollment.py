import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import silhouette_score, r2_score, mean_absolute_error, mean_squared_error
import glob, os, json, time, warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# =============================================================================
# AI LOGGING ENGINE
# =============================================================================

def log_step(step_name):
    print(f"\n⏳ {step_name}...")
    return time.time()

def end_step(start_time):
    print(f"✅ Completed in {time.time()-start_time:.2f}s")

def log_plot(name):
    print(f"📊 Generating: {name}")
    return time.time()

def end_plot(start, name):
    print(f"✅ Saved {name} ({time.time()-start:.2f}s)")

# =============================================================================
# AI OUTPUT PLATFORM
# =============================================================================

BASE_DIR = "ai_outputs"
DATASETS = f"{BASE_DIR}/datasets"
INSIGHTS = f"{BASE_DIR}/insights"
MODELS = f"{BASE_DIR}/models"
REPORTS = f"{BASE_DIR}/reports"

for folder in [DATASETS, INSIGHTS, MODELS, REPORTS]:
    os.makedirs(folder, exist_ok=True)

# =============================================================================
# PIPELINE HEADER
# =============================================================================

print("="*110)
print(" "*30 + "🏆 AADHAAR ENROLMENT ENTERPRISE AI PLATFORM 🏆")
print("="*110)

# =============================================================================
# 1. DATA INGESTION
# =============================================================================

step = log_step("AI Data Ingestion Engine")

file_pattern = "data/api_data_aadhar_enrolment/api_data_aadhar_enrolment/*.csv"
csv_files = glob.glob(file_pattern)

if not csv_files:
    raise Exception("❌ No Aadhaar enrolment files found")

dfs = []
for file in csv_files:
    df_temp = pd.read_csv(file, low_memory=False)
    dfs.append(df_temp)
    print(f"✓ Loaded {os.path.basename(file)} : {len(df_temp):,}")

raw_df = pd.concat(dfs, ignore_index=True)
print(f"\n✓ Total Records: {len(raw_df):,}")

raw_df.to_csv(f"{DATASETS}/00_raw_snapshot.csv", index=False)

end_step(step)

# =============================================================================
# 2. DATA QUALITY ENGINE
# =============================================================================

step = log_step("AI Data Quality Engine")

quality_report = {
    "total_records": len(raw_df),
    "total_columns": len(raw_df.columns),
    "missing_values": int(raw_df.isnull().sum().sum()),
    "duplicate_rows": int(raw_df.duplicated().sum()),
    "columns": list(raw_df.columns)
}

with open(f"{REPORTS}/data_quality_report.json", "w") as f:
    json.dump(quality_report, f, indent=2)

end_step(step)

# =============================================================================
# 3. DATA CLEANING + OUTLIER DETECTION
# =============================================================================

step = log_step("AI Data Cleaning & Outlier Detection")

df = raw_df.drop_duplicates()

# Parse dates
df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
df = df.dropna(subset=["date"])

# Clean text
df["state"] = df["state"].str.strip().str.title()
df["district"] = df["district"].str.strip().str.title()

# Numeric
num_cols = ["age_0_5", "age_5_17", "age_18_greater", "pincode"]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# Outlier Detection (Isolation Forest)
iso = IsolationForest(contamination=0.05, random_state=42)
outlier_features = df[["age_5_17", "age_18_greater"]]
outliers = iso.fit_predict(outlier_features)

df["is_outlier"] = outliers
outliers_df = df[df["is_outlier"] == -1]
df = df[df["is_outlier"] == 1].drop("is_outlier", axis=1)

outliers_df.to_csv(f"{DATASETS}/01_outliers_removed.csv", index=False)

df.to_csv(f"{DATASETS}/02_cleaned_data.csv", index=False)

end_step(step)

# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================

step = log_step("AI Feature Engineering")

df["total_enrollments"] = df["age_0_5"] + df["age_5_17"] + df["age_18_greater"]
df["child_enrollments"] = df["age_0_5"] + df["age_5_17"]
df["adult_enrollments"] = df["age_18_greater"]
df["child_ratio"] = df["child_enrollments"] / (df["total_enrollments"] + 1e-9)

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["quarter"] = df["date"].dt.quarter
df["day_of_week"] = df["date"].dt.dayofweek
df["week"] = df["date"].dt.isocalendar().week
df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
df["log_enrollments"] = np.log1p(df["total_enrollments"])

df.to_csv(f"{DATASETS}/03_feature_engineered.csv", index=False)

end_step(step)

# =============================================================================
# 5. TEMPORAL AI ANALYTICS
# =============================================================================

step = log_step("AI Temporal Intelligence")

daily = df.groupby("date")["total_enrollments"].sum()
monthly = df.groupby(df["date"].dt.to_period("M"))["total_enrollments"].sum()

plt.figure(figsize=(14,6))
plt.plot(daily.index, daily.values)
plt.title("Daily Aadhaar Enrolment Trend")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{INSIGHTS}/01_daily_trend.png", dpi=300)
plt.close()

plt.figure(figsize=(14,6))
plt.bar(monthly.index.astype(str), monthly.values)
plt.xticks(rotation=45)
plt.title("Monthly Aadhaar Enrolment Trend")
plt.tight_layout()
plt.savefig(f"{INSIGHTS}/02_monthly_trend.png", dpi=300)
plt.close()

end_step(step)

# =============================================================================
# 6. GEOGRAPHIC AI ANALYTICS
# =============================================================================

step = log_step("AI Geographic Intelligence")

state_summary = df.groupby("state")["total_enrollments"].agg(["sum","mean","count"]).round(2)
district_summary = df.groupby(["state","district"])["total_enrollments"].sum().reset_index()

state_summary.to_csv(f"{DATASETS}/04_state_summary.csv")
district_summary.to_csv(f"{DATASETS}/05_district_summary.csv", index=False)

plt.figure(figsize=(12,8))
state_summary.sort_values("sum", ascending=False).head(15)["sum"].plot(kind="barh")
plt.title("Top States by Enrolment")
plt.tight_layout()
plt.savefig(f"{INSIGHTS}/03_top_states.png", dpi=300)
plt.close()

end_step(step)

# =============================================================================
# 7. DEMOGRAPHIC AI ANALYTICS
# =============================================================================

step = log_step("AI Demographic Intelligence")

age_totals = [df["age_0_5"].sum(), df["age_5_17"].sum(), df["age_18_greater"].sum()]

plt.figure(figsize=(8,8))
plt.pie(age_totals, labels=["0-5","5-17","18+"], autopct="%1.1f%%")
plt.title("Age Group Distribution")
plt.tight_layout()
plt.savefig(f"{INSIGHTS}/04_age_distribution.png", dpi=300)
plt.close()

end_step(step)

# =============================================================================
# 8. AI CLUSTERING ENGINE (PCA + KMeans)
# =============================================================================

step = log_step("AI Clustering Engine")

cluster_features = df[["total_enrollments","child_ratio","age_5_17","age_18_greater"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_features)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

sil_scores = []
for k in range(2,6):
    model = MiniBatchKMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels, sample_size=5000)
    sil_scores.append(sil)
    print(f"K={k} Silhouette={sil:.4f}")

optimal_k = range(2,6)[sil_scores.index(max(sil_scores))]
print(f"🏆 Optimal Clusters: {optimal_k}")

final_model = MiniBatchKMeans(n_clusters=optimal_k, random_state=42)
df["cluster"] = final_model.fit_predict(X_pca)

df.to_csv(f"{DATASETS}/06_clustered_data.csv", index=False)

end_step(step)

# =============================================================================
# 9. AI PREDICTIVE ENGINE
# =============================================================================

step = log_step("AI Predictive Intelligence")

le = LabelEncoder()
df["state_code"] = le.fit_transform(df["state"])

features = ["year","month","day_of_week","state_code"]
X = df[features]
y = df["total_enrollments"]

split = int(len(X)*0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

rf = RandomForestRegressor(n_estimators=300, max_depth=14, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

metrics = {
    "r2": float(r2_score(y_test, y_pred)),
    "mae": float(mean_absolute_error(y_test, y_pred)),
    "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred)))
}

with open(f"{MODELS}/model_metrics.json","w") as f:
    json.dump(metrics,f,indent=2)

end_step(step)

# =============================================================================
# 10. AI ML VISUAL ANALYTICS & DIAGNOSTICS
# =============================================================================

step = log_step("AI ML Visual Analytics & Diagnostics")

# Build prediction dataframe safely
pred_df = df.loc[y_test.index].copy()

pred_df["actual"] = y_test.values
pred_df["predicted"] = y_pred
pred_df["residual"] = pred_df["actual"] - pred_df["predicted"]
pred_df["error_pct"] = (pred_df["residual"] / (pred_df["actual"] + 1)) * 100

# ------------------------------
# Plot 1: Actual vs Predicted
# ------------------------------
plt.figure(figsize=(8,8))
plt.scatter(pred_df["actual"], pred_df["predicted"], alpha=0.4)
plt.plot([pred_df["actual"].min(), pred_df["actual"].max()],
         [pred_df["actual"].min(), pred_df["actual"].max()], "r--")
plt.xlabel("Actual Enrolments")
plt.ylabel("Predicted Enrolments")
plt.title("ML Model: Actual vs Predicted")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{INSIGHTS}/ml_actual_vs_predicted.png", dpi=300)
plt.close()

# ------------------------------
# Plot 2: Residual Distribution
# ------------------------------
plt.figure(figsize=(10,6))
plt.hist(pred_df["residual"], bins=50, alpha=0.7)
plt.xlabel("Residual Error")
plt.ylabel("Frequency")
plt.title("ML Residual Distribution")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{INSIGHTS}/ml_residual_distribution.png", dpi=300)
plt.close()

# ------------------------------
# Plot 3: Residual Trend Over Time
# ------------------------------
plt.figure(figsize=(14,6))
plt.plot(pred_df["date"], pred_df["residual"], alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Date")
plt.ylabel("Residual Error")
plt.title("Residual Trend Over Time")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{INSIGHTS}/ml_residual_trend.png", dpi=300)
plt.close()

# ------------------------------
# Plot 4: Error Percentage Distribution
# ------------------------------
plt.figure(figsize=(10,6))
plt.hist(pred_df["error_pct"], bins=50, alpha=0.7)
plt.xlabel("Prediction Error %")
plt.ylabel("Frequency")
plt.title("Prediction Error Percentage Distribution")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{INSIGHTS}/ml_error_percentage.png", dpi=300)
plt.close()

# ------------------------------
# Plot 5: Feature Importance
# ------------------------------
feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

feature_importance.to_csv(f"{MODELS}/feature_importance.csv", index=False)

plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=feature_importance)
plt.title("ML Feature Importance")
plt.tight_layout()
plt.savefig(f"{INSIGHTS}/ml_feature_importance.png", dpi=300)
plt.close()

end_step(step)

# =============================================================================
# 11. AI TREND RECOGNITION ENGINE
# =============================================================================

step = log_step("AI Trend Recognition Engine")

# Daily trend
daily_trend = df.groupby("date")["total_enrollments"].sum().sort_index()

# Rolling average (weekly smoothing)
rolling_avg = daily_trend.rolling(7).mean()

# Overall trend direction
trend_direction = "Increasing" if daily_trend.iloc[-1] > daily_trend.iloc[0] else "Decreasing"

# Monthly seasonal pattern
monthly_pattern = df.groupby("month")["total_enrollments"].mean()

# Fastest growing states
state_growth = df.groupby(["state", "year"])["total_enrollments"].sum().reset_index()
state_growth["growth"] = state_growth.groupby("state")["total_enrollments"].diff()

top_growth_states = (
    state_growth.sort_values("growth", ascending=False)
    .dropna()
    .head(5)
)

# Anomaly detection using Z-score
z_scores = np.abs(stats.zscore(daily_trend))
anomaly_days = daily_trend[z_scores > 3]

# Trend report
trend_report = {
    "overall_trend": trend_direction,
    "peak_day": str(daily_trend.idxmax().date()),
    "lowest_day": str(daily_trend.idxmin().date()),
    "seasonal_peak_month": int(monthly_pattern.idxmax()),
    "fastest_growing_states": top_growth_states["state"].tolist(),
    "anomaly_days": [str(d.date()) for d in anomaly_days.index]
}

# Save trend report
with open(f"{REPORTS}/ai_trend_report.json", "w") as f:
    json.dump(trend_report, f, indent=2)

end_step(step)

# =============================================================================
# 12. AI INSIGHT ENGINE
# =============================================================================

step = log_step("AI Insight Generation Engine")

insights = {
    "dataset_size": int(len(df)),
    "date_range": f"{df['date'].min().date()} to {df['date'].max().date()}",
    "total_enrollments": int(df["total_enrollments"].sum()),
    "average_daily_enrollments": int(daily_trend.mean()),
    "trend_direction": trend_direction,
    "model_accuracy_r2": metrics["r2"],
    "top_states": (
        df.groupby("state")["total_enrollments"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .to_dict()
    ),
    "seasonal_pattern": monthly_pattern.round(0).to_dict()
}

# Save insights
with open(f"{REPORTS}/ai_insights.json", "w") as f:
    json.dump(insights, f, indent=2)

end_step(step)




# =============================================================================
# 10. EXECUTIVE REPORT
# =============================================================================

step = log_step("Generating Executive AI Report")

summary = {
    "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "total_records": int(len(df)),
    "total_enrollments": int(df["total_enrollments"].sum()),
    "total_states": int(df["state"].nunique()),
    "model_r2": metrics["r2"]
}

with open(f"{REPORTS}/executive_summary.json","w") as f:
    json.dump(summary,f,indent=2)

end_step(step)

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*110)
print("🏆 ENTERPRISE AI PLATFORM EXECUTION COMPLETE")
print("="*110)

print(f"""
📊 RECORDS ANALYZED : {len(df):,}
📈 TOTAL ENROLMENTS: {df['total_enrollments'].sum():,}
🧠 AI MODEL R²     : {metrics['r2']:.4f}

📁 OUTPUT PLATFORM:
   ai_outputs/
     ├── datasets/
     ├── insights/
     ├── models/
     └── reports/

🚀 Aadhaar Enrolment Enterprise AI Platform Ready
""")

print("="*110)
