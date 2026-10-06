import os
import opendatasets as od


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "data"
)


def download_busi_dataset():
    """
    Download the BUSI dataset into the project's data directory
    if it is not already present.
    """

    os.makedirs(
        DATASET_DIR,
        exist_ok=True
    )

    existing_items = os.listdir(
        DATASET_DIR
    )

    if existing_items:
        print(
            "The project's 'data' directory already contains files."
        )
        return

    print(
        "Downloading the BUSI dataset from Kaggle..."
    )

    dataset_url = (
        "https://www.kaggle.com/datasets/"
        "aryashah2k/breast-ultrasound-images-dataset"
    )

    try:
        od.download(
            dataset_url,
            data_dir=DATASET_DIR
        )

        print(
            "Download completed successfully!"
        )

    except Exception as e:
        print(
            f"Error during download: {e}"
        )
        print(
            "Make sure your Kaggle credentials are configured if required."
        )


if __name__ == "__main__":
    download_busi_dataset()
