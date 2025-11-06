import os
import gdown
import zipfile

def setup_models():
    models_dir = "models_output"
    zip_file = "models_output.zip"

    if os.path.exists(models_dir):
        print("✅ Models folder already exists — skipping download.")
        return

    FILE_ID = "1Hz4UFpIbNdwhSJZlLUIxCIP4OMxnzKWJ"  # 👈 only this part
    url = f"https://drive.google.com/uc?id={FILE_ID}"

    print("⬇️ Downloading models_output.zip from Google Drive...")
    gdown.download(url, zip_file, quiet=False)

    print("📦 Extracting models_output.zip ...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(".")
    
    print("✅ Extraction complete! Models ready to use.")

setup_models()
