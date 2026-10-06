# 🏥 Medical CBIR System — Breast Ultrasound Image Retrieval

A Python-based **Content-Based Image Retrieval (CBIR)** project for breast ultrasound images using the **BUSI (Breast Ultrasound Images)** dataset.

The project explores how classical image descriptors can be used to represent medical images numerically and retrieve visually similar images from a reference database.

> **Academic scope:** This is an image-retrieval and similarity-search project. It is **not a clinical diagnostic system** and does not perform medical diagnosis.

---

## 🎯 Project Objective

The objective is to build and evaluate a reproducible CBIR pipeline that:

- loads breast ultrasound images from the BUSI dataset;
- preprocesses images into a consistent representation;
- extracts complementary visual descriptors;
- normalizes the resulting feature vectors;
- retrieves the most similar images using distance-based search;
- evaluates retrieval consistency using **Precision@1** and **Precision@5**.

The project focuses on **classical feature engineering for medical image retrieval**, providing a practical foundation for understanding image similarity before moving toward more advanced deep-learning approaches.

---

## 🔬 Retrieval Pipeline

~~~text
BUSI Ultrasound Images
        │
        ▼
Dataset Loading
        │
        ▼
Preprocessing
  • Ignore segmentation masks
  • Grayscale conversion
  • Resize to 128 × 128
        │
        ▼
Feature Extraction
  ├── Intensity histogram
  ├── GLCM texture features
  └── Hu moments
        │
        ▼
Feature Normalization
      MinMaxScaler
        │
        ▼
Retrieval Database
        │
        ▼
Similarity Search
   Euclidean distance
        │
        ▼
Top-K Retrieved Images
        │
        ▼
Evaluation
  • Precision@1
  • Precision@5
  • Per-class analysis
~~~

---

## 🧠 Feature Extraction

The retrieval representation combines three complementary descriptor families implemented in **src/cbir_skimage.py**.

### 1. Intensity distribution

A **32-bin grayscale intensity histogram** is extracted from each preprocessed image.

This captures the overall distribution of pixel intensities in the ultrasound image.

### 2. Texture

Texture information is represented using a **Gray-Level Co-occurrence Matrix (GLCM)**.

The implementation extracts:

- Contrast
- Energy
- Homogeneity
- Correlation

These descriptors provide information about local intensity relationships and image texture.

### 3. Shape

The project also computes **seven Hu moments** from the largest connected region obtained after thresholding the image.

Hu moments provide compact shape-related descriptors that are relatively robust to common geometric transformations.

> These descriptors are used as image-representation features for retrieval. They are **not clinical biomarkers**.

---

## ⚙️ Preprocessing

Before feature extraction, the images are processed consistently:

1. BUSI segmentation-mask files containing _mask are excluded.
2. RGB/RGBA images are converted to grayscale when necessary.
3. Images are resized to **128 × 128 pixels** using anti-aliasing.
4. Feature vectors are generated from the processed images.
5. Features from the retrieval database are normalized using **MinMaxScaler**.

During evaluation, the scaler is fitted **only on the retrieval database** and then applied to query features. This keeps the query images independent from the feature-normalization step.

---

## 🔎 Similarity Search

The **IndexeurCBIR** class provides distance-based retrieval.

Supported distance metrics:

- **Euclidean distance** — used for the reported evaluation
- **Cosine distance** — implemented as an additional retrieval option

For each query, the system ranks database images by ascending distance and returns the requested top-K results.

The evaluation uses:

~~~text
Top-K = 5
Distance = Euclidean
~~~

The query image itself is excluded from the retrieval database.

---

## 📊 Evaluation Protocol

The evaluation script in **src/cbir_main.py** uses a balanced query set:

| Evaluation element | Configuration |
|---|---|
| Dataset | BUSI |
| Classes | Benign, Malignant, Normal |
| Query images | 30 |
| Queries per class | 10 |
| Retrieval database | Remaining BUSI images |
| Retrieval size | Top-5 |
| Distance metric | Euclidean |
| Relevance criterion | Same BUSI class as query |
| Metrics | Precision@1, Precision@5 |
| Random seed | 42 |

### Why Precision@K?

For this project, a retrieved image is considered **relevant** when it belongs to the same BUSI class as the query.

- **Precision@1**: whether the first retrieved image has the same class as the query.
- **Precision@5**: proportion of the five retrieved images that have the same class as the query.

These metrics evaluate **retrieval consistency**, not diagnostic performance.

---

## 📈 Results

The current evaluation produced:

| Metric | Result |
|---|---:|
| **Mean Precision@1** | **66.7%** |
| **Mean Precision@5** | **48.7%** |

### Results by BUSI class

| Class | Precision@1 | Precision@5 |
|---|---:|---:|
| Benign | 70.0% | 60.0% |
| Malignant | 60.0% | 44.0% |
| Normal | 70.0% | 42.0% |

The evaluation is based on **30 balanced queries (10 per class)**, with each query removed from the retrieval database.

### Interpretation

The results show that the handcrafted feature representation can retrieve images sharing the query's BUSI class with moderate consistency.

The difference between Precision@1 and Precision@5 also illustrates that the quality of the retrieved neighborhood can vary across queries. This provides a useful baseline for future experimentation with improved feature representations or learned embeddings.

---

## 📁 Project Structure

~~~text
Medical_CBIR_System/
│
├── src/
│   ├── cbir_main.py
│   ├── cbir_skimage.py
│   └── telecharger_data.py
│
├── reports/
│   ├── metrics_summary.txt
│   ├── precision_by_class.png
│   └── resultat_cbir.png
│
├── .gitignore
├── README.md
└── requirements.txt
~~~

### Main components

**src/cbir_skimage.py**

Contains the core CBIR implementation:

- BUSI image loading
- preprocessing
- feature extraction
- feature normalization
- database indexing
- similarity search
- retrieval visualization

**src/cbir_main.py**

Contains the evaluation pipeline:

- balanced query selection
- database/query separation
- retrieval evaluation
- Precision@1 and Precision@5 computation
- per-class analysis
- report generation

**src/telecharger_data.py**

Provides an optional helper for downloading the BUSI dataset from Kaggle using **opendatasets**.

**reports/**

Contains the generated evaluation outputs and visual results.

---

## 🛠️ Technologies

- **Python**
- **NumPy**
- **scikit-image**
- **scikit-learn**
- **Matplotlib**
- **OpenCV**
- **Kaggle dataset tooling**

### Main technical concepts

- Medical image preprocessing
- Content-Based Image Retrieval (CBIR)
- Feature engineering
- Intensity histograms
- GLCM texture analysis
- Hu moments
- Feature normalization
- Euclidean and cosine similarity
- Precision@K evaluation
- Reproducible experiments

---

## 🚀 Installation

Clone the repository:

~~~bash
git clone https://github.com/ghada-59/Medical_CBIR_System.git
cd Medical_CBIR_System
~~~

Install the required Python packages:

~~~bash
pip install -r requirements.txt
~~~

### Dataset

The BUSI dataset is required locally but is intentionally **not included in this repository**.

The repository's **.gitignore** excludes the dataset directories to avoid committing large dataset files.

The dataset can be obtained from the Kaggle source used by the project:

~~~text
https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
~~~

After downloading, place the dataset so that the expected structure is available under:

~~~text
data/
└── breast-ultrasound-images-dataset/
    └── Dataset_BUSI_with_GT/
        ├── benign/
        ├── malignant/
        └── normal/
~~~

---

## ▶️ Running the Evaluation

From the project root:

~~~bash
cd src
python cbir_main.py
~~~

The script will:

1. load the BUSI images;
2. select 10 queries per class;
3. create a retrieval database from the remaining images;
4. extract and normalize descriptors;
5. retrieve the five nearest images for each query;
6. calculate Precision@1 and Precision@5;
7. generate evaluation reports.

Generated outputs are saved under:

~~~text
reports/
~~~

---

## 📄 Generated Reports

### metrics_summary.txt

Contains:

- evaluation configuration;
- database/query sizes;
- global Precision@1 and Precision@5;
- results by BUSI class;
- interpretation of the metrics.

### precision_by_class.png

Visual comparison of Precision@1 and Precision@5 across:

- benign
- malignant
- normal

### resultat_cbir.png

Visualization of a query image and retrieved similar images with their corresponding distances.

---

## ⚠️ Limitations

This project is intentionally a **classical CBIR baseline** and has several limitations:

- The feature representation is handcrafted rather than learned.
- The evaluation uses a relatively small balanced query set of 30 images.
- Relevance is defined using BUSI dataset class labels, which does not mean that visual similarity corresponds to clinical similarity.
- The descriptors are not validated clinical biomarkers.
- No diagnostic conclusion should be drawn from the retrieved images.
- Retrieval performance may be affected by ultrasound acquisition variability, image quality and the limitations of handcrafted descriptors.

Therefore, the reported Precision@K values should be interpreted as **academic retrieval metrics for this experimental setup**, not clinical performance measures.

---

## 🔭 Possible Future Improvements

The current implementation provides a baseline that can be extended with:

- larger and more systematic evaluation sets;
- additional similarity metrics and feature-weighting strategies;
- stronger texture and shape representations;
- image augmentation and robustness analysis;
- classical dimensionality-reduction techniques;
- learned visual embeddings;
- deep-learning-based retrieval;
- more rigorous validation protocols.

These extensions would allow comparison between handcrafted CBIR methods and modern representation-learning approaches.

---

## 🎓 Skills Demonstrated

This project demonstrates practical work with:

- **Biomedical image data**
- **Medical image preprocessing**
- **Classical computer vision**
- **Feature engineering**
- **Image similarity and retrieval**
- **Python scientific computing**
- **scikit-image and scikit-learn**
- **Experimental evaluation**
- **Metric interpretation**
- **Reproducible data processing**
- **Git/GitHub project organization**

---

## 📌 Project Scope

**Project type:** Academic / Practice Project  
**Domain:** Biomedical Engineering · Medical Imaging · Computer Vision  
**Task:** Content-Based Image Retrieval  
**Dataset:** BUSI Breast Ultrasound Images  
**Approach:** Handcrafted visual descriptors + distance-based retrieval

This project is intended to demonstrate foundational skills in **medical imaging, computer vision and Python-based biomedical data processing**.
