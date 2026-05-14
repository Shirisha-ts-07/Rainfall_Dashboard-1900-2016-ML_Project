import pandas as pd

# ===================== LOAD FILES =====================
df_year = pd.read_csv("rainfall_predictions.csv")
df_dist = pd.read_csv("rainfallinIndia.csv")

# ===================== CLEAN =====================
df_year.columns = df_year.columns.str.strip()
df_dist.columns = df_dist.columns.str.strip()

# ===================== RENAME FOR MATCH =====================
df_year = df_year.rename(columns={"SUBDIVISION": "STATE_UT_NAME"})

# ===================== STANDARDIZE TEXT =====================
df_year["STATE_UT_NAME"] = df_year["STATE_UT_NAME"].str.upper().str.strip()
df_dist["STATE_UT_NAME"] = df_dist["STATE_UT_NAME"].str.upper().str.strip()

# ===================== FILTER KARNATAKA (optional) =====================
df_year = df_year[df_year["STATE_UT_NAME"] == "KARNATAKA"]
df_dist = df_dist[df_dist["STATE_UT_NAME"] == "KARNATAKA"]

# ===================== MERGE =====================
merged_df = pd.merge(
    df_dist,
    df_year,
    on="STATE_UT_NAME",
    how="left"
)

# ===================== SAVE =====================
merged_df.to_csv("combined_rainfall.csv", index=False)

print("✅ Combined file created: combined_rainfall.csv")