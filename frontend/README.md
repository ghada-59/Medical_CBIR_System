# Medical CBIR React Interface

This optional React interface provides a browser-based workspace for the existing classical image-retrieval implementation. It does not replace the Python retrieval code or the original command-line evaluation workflow.

## Requirements

- Python 3.10 or newer
- Node.js 18 or newer
- The BUSI dataset downloaded separately from its source

## 1. Install the API dependencies

From the repository root:

~~~bash
python -m pip install -r requirements-react-api.txt
~~~

## 2. Start the local API

From the repository root, in terminal 1:

~~~bash
python -m uvicorn cbir_api:app --reload --host 127.0.0.1 --port 8000
~~~

The health endpoint is http://127.0.0.1:8000/api/health. API documentation is available at http://127.0.0.1:8000/docs.

## 3. Start the React interface

In terminal 2:

~~~bash
cd frontend
npm install
npm run dev
~~~

Open the local URL printed by Vite (normally http://127.0.0.1:5173/). Keep both terminals open.

## Dataset layout

The API expects the locally downloaded dataset in this layout:

~~~text
data/
└── breast-ultrasound-images-dataset/
    └── Dataset_BUSI_with_GT/
        ├── benign/
        ├── malignant/
        └── normal/
~~~

Segmentation mask files are excluded by the existing loader. The app reports dataset counts from the local folders. It does not silently download data.

## Interface functions

- Immediate local preview after selecting a supported PNG, JPG, WEBP or BMP query image
- Configurable handcrafted descriptors: intensity histogram, GLCM texture and Hu moments
- Selectable Top-K retrieval count and Euclidean or cosine distance
- Ranked result previews, source filename, dataset-folder label and descriptor distance
- Dataset counts and the saved Precision@K evaluation summary when present
- Separate light and dark themes with project-specific plum/lilac styling

The first retrieval request builds an in-memory feature index and can take longer. The index is cached for subsequent searches in the same API process. Restarting the API clears that cache.

## Scope and limitations

This is an academic image-retrieval tool, not a diagnostic system. Dataset folder labels are not predictions for an uploaded query image. Precision@1 and Precision@5 describe retrieval consistency under the project's same-folder relevance rule, not clinical diagnostic accuracy. Uploaded images are processed locally by the API and are not intentionally stored by the interface.
