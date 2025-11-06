import os
import zipfile
import pandas as pd
import gdown  # pip install gdown

# -----------------------------
# Dataset & Model Links
# -----------------------------
dataset_folder = "preprocessed_datasets"
dataset_zip = "preprocessed_datasets.zip"
dataset_gdrive_id = "https://drive.google.com/file/d/1U6TeAu0XIAoxsV0_3H0jgOrSlXgvsy7l/view?usp=drive_link"

models_folder = "trained_models"
models_zip = "trained_models.zip"
models_gdrive_id = "https://drive.google.com/file/d/1uHqSk_4WjFZoxUw1ERBJkh2ZxaTq46MT/view?usp=drive_link"

# -----------------------------
# Function to download & unzip from Google Drive
# -----------------------------
def download_and_unzip(zip_name, folder_name, gdrive_id):
    if not os.path.exists(folder_name):
        print(f"📥 Downloading {zip_name} from Google Drive...")
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        gdown.download(url, zip_name, quiet=False)
        print(f"📦 Extracting {zip_name}...")
        with zipfile.ZipFile(zip_name, 'r') as zip_ref:
            zip_ref.extractall(folder_name)
        print(f"✅ {folder_name} ready!")

# -----------------------------
# Ensure dataset and models exist
# -----------------------------
download_and_unzip(dataset_zip, dataset_folder, dataset_gdrive_id)
download_and_unzip(models_zip, models_folder, models_gdrive_id)

# -----------------------------
# Load dataset CSV files
# -----------------------------
data_files = [
    os.path.join(dataset_folder, f)
    for f in os.listdir(dataset_folder)
    if f.endswith(".csv")
]

if not data_files:
    raise FileNotFoundError(f"❌ No CSV files found in {dataset_folder}!")

for file in data_files:
    df = pd.read_csv(file)
    print(f"\nLoaded {file} (first 2 rows):")
    print(df.head(2))

# -----------------------------
# Load trained models
# -----------------------------
model_files = [
    os.path.join(models_folder, f)
    for f in os.listdir(models_folder)
    if f.endswith(".h5") or f.endswith(".pkl")  # adjust based on your model format
]

if not model_files:
    raise FileNotFoundError(f"❌ No model files found in {models_folder}!")

print(f"\n✅ Found {len(model_files)} trained models in {models_folder}")
