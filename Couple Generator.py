import pandas as pd

# Load the CSV
csv_path = r"C:\Users\97254\OneDrive - mail.tau.ac.il\שולחן העבודה\לימודים\שנה ד\פרוייקט גמר\The_Child_Affective_Facial_Expression_(CAFE)_set\The Child Affective Facial Expression (CAFE) set.csv"
df = pd.read_csv(csv_path)

# Normalize text data
df['emotion'] = df['emotion'].str.lower().str.strip()
df['subfolder'] = df['subfolder'].str.strip()

# Prepare list to store valid (pre, post) pairs
pairs = []

# Group by subfolder to ensure matching sessions
for subfolder, group in df.groupby('subfolder'):
    # Filter group by emotion
    pre_group = group[group['emotion'].isin(['neutral', 'neutralopen'])]

    for _, pre_row in pre_group.iterrows():
        # Post must be different emotion from pre, and not the same image
        valid_post_group = group[
            (group['emotion'] != pre_row['emotion']) &
            (group['file_name'] != pre_row['file_name'])
        ]
        for _, post_row in valid_post_group.iterrows():
            # Label = 0 if both pre and post are neutral/neutralopen, else 1
            pre_neutral = pre_row['emotion'] in ['neutral', 'neutralopen']
            post_neutral = post_row['emotion'] in ['neutral', 'neutralopen']
            label = 0 if pre_neutral and post_neutral else 1

            pairs.append({
                'pre_filename': pre_row['file_name'],
                'post_filename': post_row['file_name'],
                'pre_emotion': pre_row['emotion'],
                'post_emotion': post_row['emotion'],
                'subfolder': subfolder,
                'label': label
            })

# Convert to DataFrame
pairs_df = pd.DataFrame(pairs)

# Optionally save to CSV
pairs_df.to_csv('emotion_image_pairs.csv', index=False)
print("Pairing complete. Saved to 'emotion_image_pairs.csv'.")
