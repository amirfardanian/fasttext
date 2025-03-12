import pandas as pd

# Load dataset
df = pd.read_csv("capture_model_old-master/capture_model/package_data/lookup_data.csv")

# Load classification function
from capture_model.scoring import classify_lines

# Extract test lines
test_lines = df["recp_line"].tolist()

# Run classification
classified_results = classify_lines(test_lines)

# Convert results into DataFrame
predictions_df = pd.DataFrame(classified_results)

# Convert Labels to Strings
df["true_label"] = df["label"].astype(str)
predictions_df["predicted_label"] = predictions_df["prediction"].astype(str)

# 📌 **1. Check Data Leakage (If Test Data Exists in Training Data)**
unique_test_lines = set(df["recp_line"])
unique_training_lines = set(df["recp_line"])  # Assuming training data is the same file (adjust if needed)

common_lines = unique_test_lines.intersection(unique_training_lines)
leakage_percentage = (len(common_lines) / len(unique_test_lines)) * 100

print(f"🔍 **Data Leakage Check:** {len(common_lines)} / {len(unique_test_lines)} test samples exist in training data.")
print(f"📊 **Percentage of leaked test samples:** {leakage_percentage:.2f}%")

if leakage_percentage > 5:
    print("⚠️ **WARNING:** More than 5% of test data exists in training data! This may cause overfitting.")

# 📌 **2. Compute Percentage of FastText vs. LookUp Predictions**
total_predictions = len(predictions_df)
lookup_predictions = sum(predictions_df["model_used"] == "model_lookup_20180301")
fasttext_predictions = sum(predictions_df["model_used"] == "model_fasttext_20180301")

lookup_percentage = (lookup_predictions / total_predictions) * 100
fasttext_percentage = (fasttext_predictions / total_predictions) * 100

print(f"\n📊 **Prediction Distribution:**")
print(f"✅ **LookUp Model Predictions:** {lookup_predictions} ({lookup_percentage:.2f}%)")
print(f"✅ **FastText Model Predictions:** {fasttext_predictions} ({fasttext_percentage:.2f}%)")

if lookup_percentage > 95:
    print("⚠️ **WARNING:** Lookup Model is predicting more than 95% of the data! It might be overfitting to memorized data.")

if fasttext_percentage > 95:
    print("⚠️ **WARNING:** FastText is predicting more than 95% of the data! Check if Lookup Model is being underused.")

