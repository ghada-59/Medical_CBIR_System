# 🏥 Medical CBIR System — Breast Ultrasound Image Retrieval

A Python project exploring **Content-Based Image Retrieval (CBIR)** for breast ultrasound images using the **BUSI dataset**.

## 🎯 Project objective

The goal is to represent medical images with numerical descriptors and retrieve visually similar images from a dataset.

This is an **academic image-retrieval project**, not a clinical diagnostic system.

## 🧠 Feature extraction

The project combines three descriptor families:

- **Intensity:** grayscale intensity histogram
- **Texture:** GLCM features such as contrast, energy, homogeneity and correlation
- **Shape:** Hu moments

The descriptors are combined into a feature vector and normalized before similarity search.

## ⚙️ Pipeline

1. Load BUSI images and exclude segmentation-mask files.
2. Convert images to grayscale and resize them to 128 × 128.
3. Extract intensity, texture and shape descriptors.
4. Normalize the feature vectors with \`MinMaxScaler\`.
5. Compare a query image with indexed images using Euclidean or cosine similarity.
6. Retrieve the most similar images.

## 📂 Structure

\`\`\`text
Medical_CBIR_System/
├── src/
│   ├── cbir_main.py
│   ├── cbir_skimage.py
│   └── telecharger_data.py
├── .gitignore
├── README.md
└── requirements.txt
\`\`\`

## 🛠️ Technologies

Python · NumPy · scikit-image · scikit-learn · Matplotlib · BUSI dataset

## ⚠️ Scope and limitations

This project demonstrates classical feature engineering and image retrieval. Similarity between images does **not** establish a diagnosis, and the extracted descriptors are not validated clinical biomarkers.

## 🚀 Run

\`\`\`bash
pip install -r requirements.txt
cd src
python cbir_main.py
\`\`\`
