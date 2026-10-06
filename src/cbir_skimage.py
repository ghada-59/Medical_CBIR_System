import os
import numpy as np
from skimage import io, color, feature, measure, transform
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def charger_images_dossier(chemin_dossier, max_images=None, random_seed=42):
    """
    Load BUSI ultrasound images from all three classes.

    - benign
    - malignant
    - normal

    Mask files (_mask) are ignored.
    Images are converted to grayscale and resized to 128x128.

    If max_images is None, all available images are loaded.
    If max_images is specified, the same number of images is
    selected from each class.

    A fixed random seed ensures reproducible selection.
    """

    import random

    images = []
    file_names = []

    classes = ['benign', 'malignant', 'normal']

    rng = random.Random(random_seed)

    for class_name in classes:

        class_dir = os.path.join(
            chemin_dossier,
            'breast-ultrasound-images-dataset',
            'Dataset_BUSI_with_GT',
            class_name
        )

        if not os.path.exists(class_dir):
            print(f"WARNING: Class folder not found: {class_dir}")
            continue

        # Find real ultrasound images and ignore segmentation masks
        class_images = [
            file for file in os.listdir(class_dir)
            if (
                file.lower().endswith(('.png', '.jpg', '.jpeg'))
                and '_mask' not in file.lower()
            )
        ]

        # Sort first for reproducibility
        class_images.sort()

        # Shuffle with a fixed seed
        rng.shuffle(class_images)

        # Select images
        if max_images is None:
            selected_files = class_images
        else:
            selected_files = class_images[:min(max_images, len(class_images))]

        print(
            f"Loading class '{class_name}': "
            f"{len(selected_files)} / {len(class_images)} images"
        )

        for file in selected_files:

            full_path = os.path.join(class_dir, file)

            try:
                img = io.imread(full_path)

                # Convert RGB/RGBA images to grayscale
                if img.ndim == 3:
                    img = color.rgb2gray(img)

                # Resize to 128x128
                img_resized = transform.resize(
                    img,
                    (128, 128),
                    anti_aliasing=True
                )

                images.append(img_resized)

                # Keep the class information in the path
                rel_path = os.path.relpath(
                    full_path,
                    os.path.dirname(chemin_dossier)
                )

                file_names.append(rel_path)

            except Exception as e:
                print(f"WARNING: Error loading {file}: {e}")

    print("\nClass-balanced loading complete.")
    print(f"Total images loaded: {len(images)}")

    return images, file_names


class IndexeurCBIR:
    def __init__(self):
        self.features_db = None
        self.file_names = None
        self.images_db = None
        self.scaler = MinMaxScaler()
        
    def extraire_caracteristiques(self, img, methodes=('intensite', 'texture', 'forme')):
        """Extract intensity, texture and shape descriptors."""
        descriptors = []
        img_uint8 = (img * 255).astype(np.uint8)
        
        # 1. Intensity histogram
        if 'intensite' in methodes:
            hist, _ = np.histogram(img_uint8, bins=32, range=(0, 256), density=True)
            descriptors.extend(hist)
            
        # 2. Texture (GLCM)
        if 'texture' in methodes:
            glcm = feature.graycomatrix(img_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
            energy = feature.graycoprops(glcm, 'energy')[0, 0]
            homogeneity = feature.graycoprops(glcm, 'homogeneity')[0, 0]
            correlation = feature.graycoprops(glcm, 'correlation')[0, 0]
            descriptors.extend([contrast, energy, homogeneity, correlation])
            
        # 3. Shape (Hu Moments)
        if 'forme' in methodes:
            thresh = img > np.mean(img)
            label_img = measure.label(thresh)
            regions = measure.regionprops(label_img)
            if regions:
                r = max(regions, key=lambda x: x.area)
                hu = r.moments_hu
            else:
                hu = np.zeros(7)
            descriptors.extend(hu)
            
        return np.array(descriptors, dtype=float)

    def indexer(self, images, noms_fichiers, methodes=('couleur', 'texture', 'forme')):
        """Indexes the image database and normalizes features."""
        self.images_db = images
        self.file_names = noms_fichiers
        features_list = []
        
        for img in images:
            feat = self.extraire_caracteristiques(img, methodes)
            features_list.append(feat)
            
        self.features_db = self.scaler.fit_transform(features_list)
        print(f"✅ Successful indexing ({len(images)} images indexed).")

    def rechercher(
    self,
    image_requete,
    top_k=5,
    metrique='euclidienne',
    methodes=('couleur', 'texture', 'forme'),
    exclude_index=None
    ):
        """Searches for the most similar images based on distance."""
        feat_req = self.extraire_caracteristiques(image_requete, methodes)
        feat_req_norm = self.scaler.transform([feat_req])[0]
        
        distances = []
        for i, feat_db in enumerate(self.features_db):
            if exclude_index is not None and i == exclude_index:
                continue
            if metrique == 'euclidienne':
                dist = np.linalg.norm(feat_req_norm - feat_db)
            elif metrique == 'cosinus':
                dot_product = np.dot(feat_req_norm, feat_db)
                norm_a = np.linalg.norm(feat_req_norm)
                norm_b = np.linalg.norm(feat_db)
                dist = 1 - (dot_product / (norm_a * norm_b + 1e-8))
            else:
                dist = np.linalg.norm(feat_req_norm - feat_db)
            distances.append((i, dist))
            
        # Sort by ascending distance
        distances.sort(key=lambda x: x[1])
        top_results = distances[:top_k]
        
        indices = [res[0] for res in top_results]
        scores = [res[1] for res in top_results]
        return indices, scores

    def afficher_resultats(self, image_requete, indices_resultats, distances):
        """Displays the query, results, and saves them to the reports folder."""
        fig, axes = plt.subplots(1, len(indices_resultats) + 1, figsize=(15, 4))
        
        # Display query image
        axes[0].imshow(image_requete, cmap='gray')
        axes[0].set_title("Query\n(Reference)", fontsize=9, fontweight='bold', color='blue')
        axes[0].axis('off')
        
        # Display similar results
        for idx, (img_idx, dist) in enumerate(zip(indices_resultats, distances)):
            ax = axes[idx + 1]
            similar_img = self.images_db[img_idx]
            ax.imshow(similar_img, cmap='gray')
            ax.set_title(f"Top {idx+1}\nDist: {dist:.3f}", fontsize=8)
            ax.axis('off')
            
        plt.tight_layout()
        
        # Automatic management of the reports folder at the project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        reports_dir = os.path.join(project_root, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        file_path = os.path.join(reports_dir, "resultat_cbir.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"💾 Visual report successfully saved: {file_path}")
        
        plt.show()