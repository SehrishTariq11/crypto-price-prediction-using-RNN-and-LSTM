import os, zipfile, gdown, streamlit as st

def setup_models():
    zip_file = "models_output.zip"
    models_dir = "models_output"
    FILE_ID = "1Hz4UFpIbNdwhSJZlLUIxCIP4OMxnzKWJ"  # your Drive ID
    url = f"https://drive.google.com/uc?id={FILE_ID}"

    if os.path.exists(models_dir):
        st.success("✅ Models already available!")
        return True

    # Download
    st.info("⬇️ Downloading trained models from Google Drive...")
    gdown.download(url, zip_file, quiet=False)

    # Extract
    try:
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(".")
        st.success("✅ ZIP extracted successfully!")

        # --- Auto-detect main folder ---
        folders = [f for f in os.listdir(".") if os.path.isdir(f)]
        found = False
        for f in folders:
            if "model" in f.lower():  # detect any folder like models_output, crypto_models, etc.
                if f != models_dir:
                    os.rename(f, models_dir)
                found = True
                break

        if not found:
            st.error("❌ 'models_output' folder not found even after extraction — check ZIP structure.")
            return False

        st.success("✅ Models ready to use!")
        return True

    except zipfile.BadZipFile:
        st.error("❌ ZIP file is invalid or corrupted.")
        return False
