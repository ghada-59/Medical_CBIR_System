import os
import numpy as np
from skimage import io, color, feature, measure, transform
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def charger_images_dossier(chemin_dossier, max_images=20):
    """
    Loads images from the dataset while ignoring masks (_mask)
    and converts them to resized grayscale arrays.
    """
    images = []
    file_names = []
    
    for root, dirs, files in os.walk(chemin_dossier):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')) and '_mask' not in file.lower():
                full_path = os.path.join(root, file)
                try:
                    img = io.imread(full_path)
                    if img.ndim == 3:
                        img = color.rgb2gray(img)
                    # Uniform resizing
                    img_resized = transform.resize(img, (128, 128), anti_aliasing=True)
                    images.append(img_resized)
                    
                    rel_path = os.path.relpath(full_path, os.path.dirname(chemin_dossier))
                    file_names.append(rel_path)
                    
                    if len(images) >= max_images:
                        return images, file_names
                except Exception as e:
                    print(f"⚠️ Error loading {file}: {e}")
                    
    return images, file_names


class IndexeurCBIR:
    def __init__(self):
        self.features_db = None
        self.file_names = None
        self.images_db = None
        self.scaler = MinMaxScaler()
        
    def extraire_caracteristiques(self, img, methodes=('couleur', 'texture', 'forme')):
        """Extracts descriptors (Intensity Histogram, GLCM, Hu Moments)."""
        descriptors = []
        img_uint8 = (img * 255).astype(np.uint8)
        
        # 1. Color (Intensity Histogram)
        if 'couleur' in methodes:
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

    def rechercher(self, image_requete, top_k=5, metrique='euclidienne', methodes=('couleur', 'texture', 'forme')):
        """Searches for the most similar images based on distance."""
        feat_req = self.extraire_caracteristiques(image_requete, methodes)
        feat_req_norm = self.scaler.transform([feat_req])[0]
        
        distances = []
        for i, feat_db in enumerate(self.features_db):
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
        reports_dir = "../reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        file_path = os.path.join(reports_dir, "resultat_cbir.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"💾 Visual report successfully saved: {file_path}")
        
        plt.show()