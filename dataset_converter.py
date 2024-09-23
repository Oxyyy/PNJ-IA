import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset 

# Charger le dataset depuis HuggingFace
def load_huggingface_dataset(dataset_name, split, text_field): 
    dataset = load_dataset(dataset_name, split=split)
    texts = dataset[text_field]  
    return texts

# Transformer le dataset en embeddings
def compute_embeddings(texts, model): #Amélioration possible avec l'utilisation d'un batch_size mais pour l'instant ignoré car nécessite d'être calculé selon la taille de l'embedding + Risque d'erreur d'allocation
    embeddings = model.encode(texts, convert_to_tensor=False)
    return np.array(embeddings)

# Créer un index FAISS
def create_faiss_index(embeddings, dimension):
    index = faiss.IndexFlatL2(dimension)  # Index pour la recherche L2 (euclidienne)
    index.add(embeddings)
    return index

# Sauvegarder l'index FAISS
def save_faiss_index(index, index_file):
    faiss.write_index(index, index_file)

# Main
if __name__ == "__main__":
    # /!\les champs suivants sont case sensitive, la vérification des champs sur Face Hugg est importante /!\
    dataset_name = "dprashar/npc_dialogue_rpg_quests"  #donnez le nom du dataset. exemple ici : dprashar/npc_dialogue_rpg_quests
    faiss_index_file = "faiss_index.index" #nom du fichier faiss qui va contenir les vecteurs
    text_field = "Text"  # Champ contenant le texte cible du dataset. ici text mais on aurait pu choisir Objective ou Title qui sont des champs correctes pour ce dataset 
    split = "train"  # Choix du split. Ici seulement train est possible
    
    # Charger le modèle SentenceTransformer
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # Modèle d'embedding léger
    
    # Charger et traiter le dataset depuis Hugging Face
    texts = load_huggingface_dataset(dataset_name, split, text_field)
    
    # Calculer les embeddings
    embeddings = compute_embeddings(texts, model)
    dimension= 768
    # Créer et sauvegarder l'index FAISS
    index = create_faiss_index(embeddings,dimension)
    save_faiss_index(index, faiss_index_file)
    
    print(f"Index FAISS sauvegardé dans {faiss_index_file}")
