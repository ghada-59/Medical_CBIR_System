import os
import opendatasets as od

def telecharger_dataset_busi():
    """
    Télécharge automatiquement le dataset BUSI depuis Kaggle 
    si le dossier ./data n'existe pas.
    """
    dossier_dataset = "./data"
    
    if os.path.exists(dossier_dataset) and os.listdir(dossier_dataset):
        print("✅ Le dossier './data' existe déjà.")
        return
    
    print("📥 Téléchargement du dataset BUSI depuis Kaggle...")
    # Lien public du dataset BUSI
    dataset_url = "https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset"
    
    try:
        od.download(dataset_url)
        print("✅ Téléchargement terminé avec succès !")
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement : {e}")
        print("💡 Assurez-vous d'avoir configuré vos identifiants Kaggle si nécessaire.")

if __name__ == "__main__":
    telecharger_dataset_busi()