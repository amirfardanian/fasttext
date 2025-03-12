import pandas as pd
from capture_model.scoring import classify_lines
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

#  Load dataset
df = pd.read_csv("capture_model_old-master/capture_model/package_data/lookup_data.csv")

# Ensure correct column names (change if needed)
df = df.rename(columns={'label': 'true_label', 'recp_line': 'text'})

# Convert true labels to string for comparison with predictions
df['true_label'] = df['true_label'].astype(str)

# Run classification on full dataset
results = classify_lines(df['text'].tolist())

# convert results to DataFrame
predictions_df = pd.DataFrame(results)

# Extract predicted labels and convert to string
predictions_df['predicted_label'] = predictions_df['prediction'].astype(str)

# Merge true labels with predictions
df['predicted_label'] = predictions_df['predicted_label']

# ompute classification performance
accuracy = accuracy_score(df['true_label'], df['predicted_label'])
report = classification_report(df['true_label'], df['predicted_label'], zero_division=0)
conf_matrix = confusion_matrix(df['true_label'], df['predicted_label'])

# Display results
print("\nAccuracy Score:", accuracy)
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", conf_matrix)


