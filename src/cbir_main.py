import os
from cbir_skimage import charger_images_dossier, IndexeurCBIR

def main():
    print("==================================================")
    print("   STARTING CBIR PROJECT - BUSI IMAGING           ")
    print("==================================================")
    
    # 1. Path to your local dataset folder
    # If you are in the src/ folder, we look for the data folder right next to it (at the root)
    DATASET_DIR = "../data"
    
    if not os.path.exists(DATASET_DIR):
        DATASET_DIR = "./data"  # Fallback if executed from the root
        
    if not os.path.exists(DATASET_DIR):
        print(f"❌ ERROR: The folder '{DATASET_DIR}' could not be found.")
        print("  Make sure your 'data' folder containing the images is present in the project.")
        return
    
    # 2. Configuration
    NUM_IMAGES = 20  # You can increase this number later
    methods = ('couleur', 'texture', 'forme')  # Feature extraction methods
    
    # 3. Loading images (subfolders benign, malignant, normal are handled automatically)
    images, file_names = charger_images_dossier(DATASET_DIR, max_images=NUM_IMAGES)
    
    if len(images) == 0:
        print("❌ No valid images found in the data folder.")
        return
        
    # 4. Indexing the image database
    indexer = IndexeurCBIR()
    indexer.indexer(images, file_names, methodes=methods)
    
    # 5. Similarity search test
    query_image = images[0]
    print(f"\n🔍 Searching for similar images to: {file_names[0]}")
    
    indices, distances = indexer.rechercher(query_image, top_k=5, metrique='euclidienne', methodes=methods)
    indexer.afficher_resultats(query_image, indices, distances)

if __name__ == "__main__":
    main()