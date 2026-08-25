# 🏥 Medical CBIR System - Breast Ultrasound Image Retrieval

This project implements a **Content-Based Image Retrieval (CBIR)** system applied to medical imaging, specifically designed for breast ultrasound analysis (the **BUSI** dataset).

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Scikit-Image-orange.svg" alt="Scikit-Image">
  <img src="https://img.shields.io/badge/Domain-Biomedical%20Engineering-green.svg" alt="Biomedical">
  <img src="https://img.shields.io/badge/Dataset-BUSI-red.svg" alt="BUSI Dataset">
</p>

---

## 🎯 Clinical Context :

Interpreting breast ultrasounds is a complex task, prone to inter-observer variability due to the noisy (speckle) nature of this type of imaging. 

In a clinical context, this system aims to serve as a **Computer-Aided Diagnosis (CAD)** tool. Rather than "blindly" classifying an image (like a standard Deep Learning black box would), the CBIR approach adopts an **Evidence-Based Medicine** philosophy. 

When a radiologist examines a new ultrasound containing a suspicious lesion, the system projects this image into a vector space to retrieve the $K$ most similar historical cases (benign or malignant). This allows the practitioner to:
1. Visually compare the new lesion with past cases where the pathology is confirmed (Ground Truth).
2. Justify their clinical decision by relying on objective mathematical similarities.

---

## 🧠 Medical and Mathematical Interpretation :

The main challenge in this biomedical data pipeline lies in Feature Extraction. We have selected three families of descriptors (Intensity, Texture, Shape) that directly translate clinical biomarkers into mathematical vectors:

### 1. Intensity (Histogram) ➔ *Tissue Echogenicity*
* **Technical approach:** Calculating the distribution of grayscale levels across 32 bins.
* **Clinical interpretation:** In ultrasound imaging, malignant tumors often appear distinctly **hypoechoic** (darker) compared to surrounding adipose or glandular tissues. The histogram captures this global acoustic impedance signature.

### 2. Texture (GLCM Matrix) ➔ *Intratumoral Heterogeneity*
* **Technical approach:** Spatial pixel analysis via the Gray-Level Co-occurrence Matrix (GLCM) to extract Contrast, Energy, Homogeneity, and Correlation.
* **Clinical interpretation:** A benign tumor (e.g., fibroadenoma) generally has homogeneous internal texture. Conversely, a malignant tumor often presents **tissue heterogeneity** (internal necrosis, microcalcifications), which translates mathematically into a drop in Energy and Homogeneity, and an increase in Contrast within the GLCM features.

### 3. Shape (Hu Moments) ➔ *Lesion Morphology and Margins*
* **Technical approach:** Binarization, connected component labeling, and calculation of the 7 Hu moments (invariant to translation, rotation, and scale).

* **Clinical interpretation:** Contour morphology is a major diagnostic criterion (BI-RADS system). Benign lesions have regular, oval, and well-circumscribed shapes. Malignant tumors often have **spiculated, microlobulated, or irregular** shapes. Hu moments robustly capture this geometric complexity, regardless of the image's size or orientation.

---

## ⚙️ Data Pipeline Architecture

The system is designed modularly to ensure robust data processing (ETL):

1. **Preprocessing (Data Ingestion):** Dynamic loading of the dataset, automatic exclusion of segmentation masks (`_mask`), conversion to grayscale, and uniform resizing ($128 \times 128$) to standardize calculations.
2. **Vectorization (Feature Engineering):** Each image is transformed into a unique numerical vector grouping intensity, texture, and morphological features.
3. **Normalization:** Using Scikit-learn's `MinMaxScaler` to scale all features to a $[0, 1]$ range. This step is crucial to prevent a descriptor with numerically large values (e.g., Histogram) from overwhelming smaller values (e.g., Hu Moments) during distance calculation.
4. **Similarity Search (Retrieval):** Calculating the Euclidean (or Cosine) distance between the query image vector and the indexed database to extract the *Top-K* clinically closest images.

---

## 📂 Project Structure

```text
Projet_CBIR_Echo/
│
├── data/                  # Dataset folder (benign, malignant, normal)
├── reports/               # Generated visual results (Top-K similarities)
├── src/                   # Main source code
│   ├── cbir_main.py       # Orchestrator script (Pipeline execution)
│   ├── cbir_skimage.py    # Extraction and indexing engine
│   └── telecharger_data.py# Automatic download script (Kagglehub)
│
├── .gitignore
├── README.md              # Project documentation
└── requirements.txt       # Environment and frozen dependencies


```

---

## 🚀 Usage

### Prerequisites

Ensure the environment is configured with the required dependencies (including NumPy 1.26.4 to guarantee scientific library compatibility):

```bash
pip install -r requirements.txt

```

### Running the Pipeline

Run the main script from the `src` folder to start indexing and generate the visual report of similar cases:

```bash
cd src
python cbir_main.py

```