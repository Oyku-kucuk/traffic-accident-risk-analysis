import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================
# 1. LOAD AND SAMPLE DATA
# =========================

df = pd.read_csv("US_Accidents_March23.csv")

df_sample = df.sample(
    n=100000,
    random_state=42
).copy()


# =========================
# 2. DATETIME FEATURE ENGINEERING
# =========================

df_sample["Start_Time"] = (
    df_sample["Start_Time"]
    .astype(str)
    .str.replace(r"\.\d+$", "", regex=True)
)

df_sample["Start_Time"] = pd.to_datetime(
    df_sample["Start_Time"],
    errors="coerce"
)

df_sample["hour"] = df_sample["Start_Time"].dt.hour
df_sample["month"] = df_sample["Start_Time"].dt.month
df_sample["day_of_week"] = df_sample["Start_Time"].dt.dayofweek


# =========================
# 3. TARGET VARIABLE
# =========================

df_sample["Risk_Level"] = df_sample["Severity"].apply(
    lambda x: 0 if x <= 2 else 1
)


# =========================
# 4. DROP UNNECESSARY COLUMNS
# =========================

columns_to_drop = [
    "ID",
    "Source",
    "Description",
    "Street",
    "Zipcode",
    "Airport_Code",
    "Weather_Timestamp",
    "End_Time",
    "End_Lat",
    "End_Lng",
    "Wind_Chill(F)",
    "Start_Time"
]

df_sample = df_sample.drop(
    columns=columns_to_drop,
    errors="ignore"
)


# =========================
# 5. MISSING VALUE HANDLING
# =========================

num_cols = df_sample.select_dtypes(
    include=["number"]
).columns

for col in num_cols:
    df_sample[col] = df_sample[col].fillna(
        df_sample[col].median()
    )

cat_cols = df_sample.select_dtypes(
    include=["object", "string"]
).columns

for col in cat_cols:
    df_sample[col] = df_sample[col].fillna("Unknown")


# =========================
# 6. WEATHER GROUPING
# =========================

def weather_group(condition):
    condition = str(condition)

    if any(x in condition for x in ["Fair", "Clear"]):
        return "Clear"
    elif any(x in condition for x in ["Cloudy", "Overcast", "Scattered Clouds"]):
        return "Cloudy"
    elif any(x in condition for x in ["Rain", "Drizzle"]):
        return "Rain"
    elif any(x in condition for x in ["Snow", "Wintry Mix"]):
        return "Snow"
    elif any(x in condition for x in ["Fog", "Haze", "Smoke"]):
        return "Fog"
    elif any(x in condition for x in ["Thunder", "T-Storm"]):
        return "Storm"
    else:
        return "Other"


df_sample["Weather_Group"] = df_sample["Weather_Condition"].apply(
    weather_group
)

# Original Weather_Condition is dropped after grouping
df_sample = df_sample.drop(
    columns=["Weather_Condition"],
    errors="ignore"
)


# =========================
# 7. FINAL DATA CHECK
# =========================

print("Final shape:", df_sample.shape)

print("\nMissing values:")
print(
    (df_sample.isnull().mean() * 100)
    .sort_values(ascending=False)
    .head(10)
)

print("\nWeather groups:")
print(df_sample["Weather_Group"].value_counts())

print("\nRisk level distribution:")
print(df_sample["Risk_Level"].value_counts(normalize=True) * 100)


# =========================
# 8. BASIC EDA VISUALIZATIONS
# =========================

plt.figure(figsize=(6, 4))
df_sample["Severity"].value_counts().sort_index().plot(kind="bar")
plt.title("Accident Severity Distribution")
plt.xlabel("Severity")
plt.ylabel("Count")
plt.show()


plt.figure(figsize=(8, 4))
df_sample["hour"].value_counts().sort_index().plot(kind="bar")
plt.title("Accidents by Hour")
plt.xlabel("Hour")
plt.ylabel("Count")
plt.show()


# =========================
# 9. CORRELATION HEATMAP
# =========================

corr = df_sample.select_dtypes(
    include=["number"]
).corr()

plt.figure(figsize=(14, 10))

sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    annot=False
)

plt.title("Correlation Heatmap of Traffic Accident Features")
plt.show()

print("\nCorrelation with Severity:")
print(corr["Severity"].sort_values(ascending=False))


# =========================
# 10. K-MEANS CLUSTERING
# =========================

cluster_df = df_sample.select_dtypes(
    include=[np.number, "bool"]
).copy()

cluster_df = cluster_df.drop(
    columns=["Severity", "Risk_Level"],
    errors="ignore"
)

cluster_df = cluster_df.fillna(
    cluster_df.median()
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_df)


# Elbow Method
wcss = []

for k in range(1, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()


# KMeans with K=4
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

cluster_df["Cluster"] = clusters

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

centers = pca.transform(kmeans.cluster_centers_)

plt.figure(figsize=(10, 7))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap="viridis",
    s=12,
    alpha=0.6
)

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    c="red",
    marker="X",
    s=250,
    label="Centroids"
)

plt.title("K-Means Clustering of Traffic Accidents (K=4)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Cluster")
plt.legend()
plt.show()


# Cluster summary
cluster_summary = cluster_df.groupby("Cluster").mean()

print("\nCluster Summary:")
print(cluster_summary)


# Cluster distribution
cluster_counts = cluster_df["Cluster"].value_counts().sort_index()

plt.figure(figsize=(7, 5))
cluster_counts.plot(kind="bar")
plt.title("Cluster Distribution")
plt.xlabel("Cluster")
plt.ylabel("Number of Accidents")
plt.show()


# =========================
# 11. RANDOM FOREST CLASSIFICATION
# =========================

model_df = pd.get_dummies(
    df_sample,
    drop_first=True
)

X = model_df.drop(
    columns=["Risk_Level", "Severity"],
    errors="ignore"
)

y = model_df["Risk_Level"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# =========================
# 12. CONFUSION MATRIX
# =========================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# =========================
# 13. FEATURE IMPORTANCE
# =========================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Feature Importances:")
print(importance.head(15))


top15 = importance.head(15)

plt.figure(figsize=(10, 7))

plt.barh(
    top15["Feature"],
    top15["Importance"]
)

plt.title("Top 15 Most Important Features")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.show()