import pandas as pd

# Encoding for labels
label_encoding = {
    'airplane': 0, 'bear': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5,
    'car': 6, 'cat': 7, 'chair': 8, 'clock': 9, 'dog': 10, 'elephant': 11,
    'keyboard': 12, 'knife': 13, 'oven': 14, 'truck': 15
}

# Load the first CSV file
df1 = pd.read_csv('../hai_baseline_model_preds_max_normalized.csv')

# Load the second CSV file
df2 = pd.read_csv('../human_only_classification_6per_img_export.csv')

# Extract relevant columns from both CSV files
df1_selected = df1[df1['model_name'] == 'resnet152'][['image_name', 'noise_type', 'noise_level'] + list(label_encoding.keys()) + ['category']]
df2_selected = df2[['image_name', 'noise_type', 'noise_level' , 'image_category', 'participant_id', 'participant_classification']]

# Merge based on the 'image_name' column
merged_df = pd.merge(df1_selected, df2_selected, how='inner', suffixes=('', ''))

# Pivot the data to have separate columns for each participant's classification
pivot_df = merged_df.pivot_table(index=['image_name', 'noise_type', 'noise_level'], columns='participant_id', values='participant_classification', aggfunc='first').reset_index()

# Add columns for each class and 'category' from the first CSV
merged_df_final = pd.merge(pivot_df, df1_selected, how='inner', suffixes=('', ''))

# Replace values in columns 1 to 195 with label encodings
merged_df_final.iloc[:, 3:198] = merged_df_final.iloc[:, 3:198].replace(label_encoding)
merged_df_final['category'] = merged_df_final['category'].replace(label_encoding)

# Count the occurrences of each classification for each row (axis=1)
counts = merged_df_final.iloc[:, 3:198].copy()
for i in range(0, 16):
    counts[str(i)] = (merged_df_final.iloc[:, 3:198] == i).T.sum()
datapoints = merged_df_final.iloc[:, 0:3].copy()
final_data = pd.merge(datapoints, counts, how='inner', right_index=True, left_index=True, suffixes=('', ''))

final_data.to_csv('resnet152_imagenet_data_counts.csv', index=False)

