import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from datasets import load_dataset 

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # Modèle d'embedding léger

# Charger le dataset depuis un fichier JSON
def load_json_dataset(json_file, text_field):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    texts = [item[text_field] for item in data]  # Extraire les textes du champ spécifié
    return texts


# Transformer le dataset en embeddings
def compute_embeddings(texts, embedding_model, batch_size=32): 
    embeddings = []
    for start_idx in range(0, len(texts), batch_size):
        batch = texts[start_idx:start_idx + batch_size]
        batch_embeddings = embedding_model.encode(batch, convert_to_tensor=False)
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# Créer un index FAISS
def create_faiss_index(embeddings, dimension=384):
    index = faiss.IndexFlatL2(dimension)  # Index pour la recherche L2 (euclidienne)
    index.add(embeddings)
    return index

# Sauvegarder l'index FAISS
def save_faiss_index(index, index_file):
    faiss.write_index(index, index_file)

# Main
if __name__ == "__main__":
    
    faiss_index_file = "faiss_index.index" #nom du fichier faiss qui va contenir les vecteurs
    text_field = "Text"  # Champ contenant le texte cible du dataset. ici text mais on aurait pu choisir Objective ou Title qui sont des champs correctes pour ce dataset
    texts = load_json_dataset
    
    # Calculer les embeddings
    embeddings = compute_embeddings(texts, embedding_model, batch_size=32)
    
    # Créer et sauvegarder l'index FAISS
    index = create_faiss_index(embeddings,dimension=384) #dimension fixé par all-MiniLM-L6-v2
    save_faiss_index(index, faiss_index_file)
    print(f"Index FAISS sauvegardé dans {faiss_index_file}")
    
    
    
    
    
    
# A mettre dans le main
# /!\les champs suivants sont case sensitive, la vérification des champs sur Face Hugg est importante /!\
#dataset_name = "dprashar/npc_dialogue_rpg_quests"  #donnez le nom du dataset. exemple ici : dprashar/npc_dialogue_rpg_quests 
#split = "train"  # Choix du split. Ici seulement train est possible
# Charger et traiter le dataset depuis Hugging Face
#texts = load_huggingface_dataset(dataset_name, split, text_field)

# Charger le dataset depuis HuggingFace
#def load_huggingface_dataset(dataset_name, split, text_field): 
#    dataset = load_dataset(dataset_name, split=split)
#    texts = dataset[text_field]  
#    return [text for text in texts if text] #sécurité pour avoir bien un rendu (return texts marche très bien aussi mais ça évite les nones ou vide)
