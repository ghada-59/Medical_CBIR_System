import os
from cbir_skimage import charger_images_dossier, IndexeurCBIR

def main():
    print("==================================================")
    print("   LANCEMENT DU PROJET CBIR - IMAGERIE BUSI       ")
    print("==================================================")
    
    # 1. Chemin vers votre dossier de données local
    # Si vous êtes dans le dossier src/, on cherche le dossier data juste à côté (à la racine)
    DOSSIER_DATASET = "../data"
    
    if not os.path.exists(DOSSIER_DATASET):
        DOSSIER_DATASET = "./data"  # Solution de repli si exécuté depuis la racine
        
    if not os.path.exists(DOSSIER_DATASET):
        print(f"❌ ERREUR : Le dossier '{DOSSIER_DATASET}' est introuvable.")
        print("   Vérifiez que votre dossier 'data' contenant les images est bien présent dans le projet.")
        return
    
    # 2. Paramétrage
    NOMBRE_IMAGES = 20  # Vous pourrez augmenter ce nombre plus tard
    methodes = ('couleur', 'texture', 'forme')
    
    # 3. Chargement des images (les sous-dossiers benign, malignant, normal sont gérés automatiquement)
    images, noms_fichiers = charger_images_dossier(DOSSIER_DATASET, max_images=NOMBRE_IMAGES)
    
    if len(images) == 0:
        print("❌ Aucune image valide trouvée dans le dossier data.")
        return
        
    # 4. Indexation de la base d'images
    indexeur = IndexeurCBIR()
    indexeur.indexer(images, noms_fichiers, methodes=methodes)
    
    # 5. Test de recherche par similarité
    image_requete = images[0]
    print(f"\n🔍 Recherche des images similaires pour : {noms_fichiers[0]}")
    
    indices, distances = indexeur.rechercher(image_requete, top_k=5, metrique='euclidienne', methodes=methodes)
    indexeur.afficher_resultats(image_requete, indices, distances)

if __name__ == "__main__":
    main()