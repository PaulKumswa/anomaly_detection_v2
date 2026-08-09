# File to copy data into appropriate folders
# File created with help from ChatGPT for syntax and structure

import pandas as pd
from pathlib import Path
import shutil

#create paths
project_root = Path(__file__).resolve().parent
raw_root = project_root / "data" / "severstal-steel-defect-detection"
csv_path = raw_root / "train.csv"
train_images_path  = raw_root / "train_images"
steel_defect_path = project_root / "data" / "steel_defect"
print(project_root)
print(raw_root)
print(csv_path)

# loading csv
df = pd.read_csv(csv_path)
print(df.shape)
print(df.columns)
print(df.head())

# keep valid rows
defective = df[df["EncodedPixels"].notna()]
print(defective.shape)
print(defective["ClassId"].value_counts().sort_index())

# find raw training images
all_image_paths = list(train_images_path.glob("*.jpg"))
all_images = {path.name for path in all_image_paths}
print(f"Images found: {len(all_images)}")

# find no-defect images
defective_images = set(defective["ImageId"].unique())
no_defect_images = all_images - defective_images
print(f"All images: {len(all_images)}")
print(f"Defective images: {len(defective_images)}")
print(f"No-defect images: {len(no_defect_images)}")

# group defect classes
labels_by_image = defective.groupby("ImageId")["ClassId"].apply(
    lambda values: tuple(sorted(values.unique()))
)
print(labels_by_image.head(20))
print(labels_by_image.apply(len).value_counts())

# put images in classes
label_counts = {
    "no_defect": 0,
    "defect_1": 0,
    "defect_2": 0,
    "defect_3": 0,
    "defect_4": 0,
}

skipped = 0

# make folders
for folder_name in label_counts:
    folder_path = steel_defect_path / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)

# sort images
for image_name in sorted(all_images):
    if image_name in no_defect_images:
        destination_folder = "no_defect"
    else:
        classes = labels_by_image.loc[image_name]

        if len(classes) == 1:
            class_id = classes[0]
            destination_folder = f"defect_{class_id}"

        else:
            skipped += 1
            continue

    label_counts[destination_folder] += 1
    source_path = train_images_path / image_name
    destination_path = steel_defect_path / destination_folder / image_name
    shutil.copy2(source_path, destination_path)

classified_total = sum(label_counts.values())

# check totals
print(label_counts)
print(f"Classified images: {classified_total}")
print(f"Multi-label images skipped: {skipped}")
print(f"Total accounted for: {classified_total + skipped}")

