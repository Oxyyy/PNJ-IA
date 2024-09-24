from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer
import torch
import sys
import faiss
import numpy as np

tokenizer = GPT2Tokenizer.from_pretrained("af1tang/personaGPT")
model = GPT2LMHeadModel.from_pretrained("af1tang/personaGPT")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#Se met en mode GPU si disponble
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def flatten(lst):
    return [item for sublist in lst for item in sublist]

#Menu du niveau de créativité
lvl = -1
while lvl not in [0, 1, 2, 3]:
    lvl = int(input("What level of creativity you want to allow your npc to be?\n1 - Creative \n2 - Normal \n3 - Precise \n0 - Quit\n"))

if lvl == 0:
    print("Exiting...")
    sys.exit()

#Niveaux
if lvl == 1:
    top_k = 50
    top_p = 0.8
    temperature = 2.0
    print("NPC set to Creative level.")
elif lvl == 2:
    top_k = 20
    top_p = 0.9
    temperature = 1.0
    print("NPC set to Normal level.")
elif lvl == 3:
    top_k = 6
    top_p = 0.95
    temperature = 0.7
    print("NPC set to Precise level.")

def generate_next(bot_input_ids, top_k, top_p,temperature,max_length=1000, do_sample=True, pad_token=tokenizer.eos_token_id):
    # Create an attention mask
    attention_mask = bot_input_ids.ne(tokenizer.pad_token_id).long()

    # Pass the attention mask to the generate function
    full_msg = model.generate(
        bot_input_ids, 
        attention_mask=attention_mask,
        do_sample=do_sample, 
        top_k=top_k, 
        top_p=top_p, 
        temperature=temperature,
        max_length=max_length, 
        pad_token_id=pad_token
    )

    msg = tokenizer.decode(full_msg[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return msg

#Utility
# Convertir en données NumPy
def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

# Convertir en variable Tensor
def to_var(x):
    if not torch.is_tensor(x):
        x = torch.Tensor(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def display_dialog_history(dialog_hx):
    for j, line in enumerate(dialog_hx):
        msg = tokenizer.decode(line,clean_up_tokenization_spaces=True)
        if j %2 == 0:
            print(">> User: "+ msg)
        else:
            print("> Your Npc: "+msg)
            print()

#FAISS
# Charger l'index FAISS
def load_faiss_index(index_file):
    index = faiss.read_index(index_file)
    return index

# Rechercher les contextes pertinents en fonction du prompt utilisateur
def search_index(index, query, embedding_model, texts, top_k=5):
     # Encode la querry pour obtenir l'embedding
    query_embedding = embedding_model.encode([query])  
    distances, indices = index.search(np.array(query_embedding), top_k)
    
    # Vérifie si les indices existent ou s'il y a une relation
    if len(indices) == 0 or len(indices[0]) == 0:
        print("No results found for the query.")
        return [], []
    
    # Récupère le texte si un résultat est trouvé
    results = [texts[i] for i in indices[0] if i < len(texts)]  # Ensure index is within bounds
    return results, distances[0]


# Main
if __name__ == "__main__":
    #Récupère le fichier
    faiss_index_file = "faiss_index.index"
    # Charger l'index FAISS
    index = load_faiss_index(faiss_index_file)

    contexts = [] 
    n_contexts = int(input("How many personality or contextual facts do you want to provide? Enter -1 for Default(3): "))
    if n_contexts == -1:
        n_contexts = 3
    for i in range(n_contexts):
        context = input(">> Context/Fact %d: " % (i + 1))
        contexts.append(context)

    dialog_hx = []  # Initialiser l'historique des dialogues

    while True:
        user_inp = input(">> User: ")
        # Recherche de contextes pertinents et combine en une seule chaîne
        results, _ = search_index(index, user_inp, embedding_model, contexts, top_k=5)
        context = ' '.join(results)

        # Encoder l'entrée de l'utilisateur avec le contexte
        if len(dialog_hx) == 0:
            full_input = ['<CONTEXT> ' + ''.join(contexts)] + ['<USER> ' + user_inp]
        else:
            full_input = flatten(['<USER> ' + user_inp] + ['<NPC> ' + response for response in dialog_hx])
    
        bot_input_ids = tokenizer.encode(''.join(full_input), return_tensors='pt').to(device)
        # Générer une réponse
        msg = generate_next(bot_input_ids, top_k, top_p, temperature)
        dialog_hx.append(msg)

        print("> Your Npc: {}".format(msg))





 #utility
#deprecated version
#flatten = lambda l: [item for sublist in l for item in sublist]

#deprecated
#if torch.cuda.is_available():
#    model = model.cuda()





#==============================================================================================#

#============================pour hardcoder des personnalités==================================#
# personas = ["My name is Rick", "I like chocolate"]
# for persona in personas:
#     persona += tokenizer.eos_token

# personas = tokenizer.encode(''.join(['<|p2|>'] + personas + ['<|sep|>'] + ['<|start|>']))
#==============================================================================================#


#==================================conversation sur X tours====================================#



#pour travailler en Local   
#tokenizer = GPT2Tokenizer.from_pretrained("./tokenizer")
#model = GPT2LMHeadModel.from_pretrained("./model")

# Sauvegarder le tokenizer et le modèle localement
#tokenizer.save_pretrained('./local_model/tokenizer')  # Sauvegarde le tokenizer
#model.save_pretrained('./local_model/model')  # Sauvegarde le modèle
