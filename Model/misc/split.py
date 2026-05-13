# split dataset
import os
import glob
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split


image_folder = "./images"
mask_folder = "./masks"
output_folder = "./multitask"
metadata_path = "GroundTruth.csv"


# load labels
df = pd.read_csv(metadata_path)
df.columns = df.columns.str.strip()


# Mapping one-hot to single label for distribution check
classes = ['MEL','NV','BCC','AKIEC','BKL','DF','VASC']
df['label'] = df[classes].idxmax(axis=1)


# split (70/15/15)
train_df, rem_df = train_test_split(
   df, test_size=0.30, random_state=42, stratify=df['label']
)
valid_df, test_df = train_test_split(
   rem_df, test_size=0.50, random_state=42, stratify=rem_df['label']
)


# folder & file copying
for split_name, split_df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
    for sub in ["images", "masks"]:
        os.makedirs(os.path.join(output_folder, split_name, sub), exist_ok=True)
    shutil.copy(metadata_path, os.path.join(output_folder, split_name, "labels.csv"))


    # copy images and masks
    for _, row in split_df.iterrows():
        img_id = row['image']
        # images
        src_img = os.path.join(image_folder, f"{img_id}.jpg")
        if os.path.exists(src_img):
            shutil.copy(src_img, os.path.join(output_folder, split_name, "images", f"{img_id}.jpg"))
        # masks
        src_mask = os.path.join(mask_folder, f"{img_id}_segmentation.png")
        if os.path.exists(src_mask):
            shutil.copy(src_mask, os.path.join(output_folder, split_name, "masks", f"{img_id}_segmentation.png"))


# SUMMARY BLOCK
print(f"Image folder exists? {os.path.exists(image_folder)}")
print(f"Mask folder exists? {os.path.exists(mask_folder)}")
print(f"Train size: {len(train_df)}")
print(f"Valid size: {len(valid_df)}")
print(f"Test size: {len(test_df)}")


for name, df_split in [("Train", train_df), ("Valid", valid_df), ("Test", test_df)]:
   print(f"\n{name} distribution:")
   print("label")
   print(df_split['label'].value_counts())


print(f"\nDone! Train: {len(train_df)} images, Valid: {len(valid_df)} images, Test: {len(test_df)} images")
