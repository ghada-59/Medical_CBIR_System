import os
import opendatasets as od

def download_busi_dataset():
    """
    Automatically downloads the BUSI dataset from Kaggle 
    if the ./data folder does not exist.
    """
    dataset_dir = "./data"
    
    if os.path.exists(dataset_dir) and os.listdir(dataset_dir):
        print("✅ The './data' folder already exists.")
        return
    
    print("📥 Downloading the BUSI dataset from Kaggle...")
    # Public link of the BUSI dataset
    dataset_url = "https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset"
    
    try:
        od.download(dataset_url)
        print("✅ Download completed successfully!")
    except Exception as e:
        print(f"❌ Error during download: {e}")
        print("💡 Make sure you have configured your Kaggle credentials if necessary.")

if __name__ == "__main__":
    download_busi_dataset()