import os
import numpy as np
from skimage import io, color, feature, measure, transform
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def charger_images_dossier(chemin_dossier, max_images=20):
    """
    Charge les images du dataset en ignorant les masques (_mask)
    et les convertit en nuances de gris redimensionnées.
    """
    images = []
    noms_fichiers = []
    
    for root, dirs, files in os.walk(chemin_dossier):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')) and '_mask' not in file.lower():
                chemin_complet = os.path.join(root, file)
                try:
                    img = io.imread(chemin_complet)
                    if img.ndim == 3:
                        img = color.rgb2gray(img)
                    # Redimensionnement uniforme
                    img_resized = transform.resize(img, (128, 128), anti_aliasing=True)
                    images.append(img_resized)
                    
                    rel_path = os.path.relpath(chemin_complet, os.path.dirname(chemin_dossier))
                    noms_fichiers.append(rel_path)
                    
                    if len(images) >= max_images:
                        return images, noms_fichiers
                except Exception as e:
                    print(f"⚠️ Erreur chargement {file}: {e}")
                    
    return images, noms_fichiers


class IndexeurCBIR:
    def __init__(self):
        self.features_db = None
        self.noms_fichiers = None
        self.images_db = None
        self.scaler = MinMaxScaler()
        
    def extraire_caracteristiques(self, img, methodes=('couleur', 'texture', 'forme')):
        """Extrait les descripteurs (Histogramme, GLCM, Moments de Hu)."""
        descripteurs = []
        img_uint8 = (img * 255).astype(np.uint8)
        
        # 1. Couleur (Histogramme d'intensité)
        if 'couleur' in methodes:
            hist, _ = np.histogram(img_uint8, bins=32, range=(0, 256), density=True)
            descripteurs.extend(hist)
            
        # 2. Texture (GLCM)
        if 'texture' in methodes:
            glcm = feature.graycomatrix(img_uint8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
            energy = feature.graycoprops(glcm, 'energy')[0, 0]
            homogeneity = feature.graycoprops(glcm, 'homogeneity')[0, 0]
            correlation = feature.graycoprops(glcm, 'correlation')[0, 0]
            descripteurs.extend([contrast, energy, homogeneity, correlation])
            
        # 3. Forme (Moments de Hu)
        if 'forme' in methodes:
            thresh = img > np.mean(img)
            label_img = measure.label(thresh)
            regions = measure.regionprops(label_img)
            if regions:
                r = max(regions, key=lambda x: x.area)
                hu = r.moments_hu
            else:
                hu = np.zeros(7)
            descripteurs.extend(hu)
            
        return np.array(descripteurs, dtype=float)

    def indexer(self, images, noms_fichiers, methodes=('couleur', 'texture', 'forme')):
        """Indexe la base d'images et normalise les caractéristiques."""
        self.images_db = images
        self.noms_fichiers = noms_fichiers
        features_list = []
        
        for img in images:
            feat = self.extraire_caracteristiques(img, methodes)
            features_list.append(feat)
        
        self.features_db = self.scaler.fit_transform(features_list)
        print(f"✅ Indexation réussie ({len(images)} images indexées).")

    def rechercher(self, image_requete, top_k=5, metrique='euclidienne', methodes=('couleur', 'texture', 'forme')):
        """Recherche les images les plus similaires par distance."""
        feat_req = self.extraire_caracteristiques(image_requete, methodes)
        feat_req_norm = self.scaler.transform([feat_req])[0]
        
        distances = []
        for i, feat_db in enumerate(self.features_db):
            if metrique == 'euclidienne':
                dist = np.linalg.norm(feat_req_norm - feat_db)
            elif metrique == 'cosinus':
                produit_scalaire = np.dot(feat_req_norm, feat_db)
                norme_a = np.linalg.norm(feat_req_norm)
                norme_b = np.linalg.norm(feat_db)
                dist = 1 - (produit_scalaire / (norme_a * norme_b + 1e-8))
            else:
                dist = np.linalg.norm(feat_req_norm - feat_db)
            distances.append((i, dist))
            
        # Trier par distance croissante
        distances.sort(key=lambda x: x[1])
        top_resultats = distances[:top_k]
        
        indices = [res[0] for res in top_resultats]
        scores = [res[1] for res in top_resultats]
        return indices, scores

    def afficher_resultats(self, image_requete, indices_resultats, distances):
        """Affiche la requête, les résultats et les sauvegarde dans le dossier reports."""
        fig, axes = plt.subplots(1, len(indices_resultats) + 1, figsize=(15, 4))
        
        # Affichage de l'image requête
        axes[0].imshow(image_requete, cmap='gray')
        axes[0].set_title("Requete\n(Reference)", fontsize=9, fontweight='bold', color='blue')
        axes[0].axis('off')
        
        # Affichage des résultats similaires
        for idx, (img_idx, dist) in enumerate(zip(indices_resultats, distances)):
            ax = axes[idx + 1]
            img_similaire = self.images_db[img_idx]
            ax.imshow(img_similaire, cmap='gray')
            ax.set_title(f"Top {idx+1}\nDist: {dist:.3f}", fontsize=8)
            ax.axis('off')
            
        plt.tight_layout()
        
        # Gestion automatique du dossier reports à la racine du projet
        dossier_reports = "../reports"
        os.makedirs(dossier_reports, exist_ok=True)
        
        chemin_fichier = os.path.join(dossier_reports, "resultat_cbir.png")
        plt.savefig(chemin_fichier, dpi=300, bbox_inches='tight')
        print(f"💾 Rapport visuel enregistré avec succès : {chemin_fichier}")
        
        plt.show()