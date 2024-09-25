from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer
import torch
import sys
import faiss
import numpy as np
import pandas as pd

tokenizer = GPT2Tokenizer.from_pretrained("af1tang/personaGPT")
model = GPT2LMHeadModel.from_pretrained("af1tang/personaGPT")
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#Se met en mode GPU si disponble
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

#Récupère le fichier et charge l'index FAISS de personnalité
faiss_index_file = "faiss_index.index"
personality_index = faiss.read_index(faiss_index_file)

# Lire le fichier CSV
df = pd.read_csv('npc-dataset.csv')
# Recréer la variable documents
documents = (df['Text']).tolist()

#Recréer la variable documents en incluant tous les champs
#documents = df[['Title', 'Objective', 'Text']].astype(str).agg(' '.join, axis=1).tolist()


#Récupère le fichier et charge l'index FAISS de réplique
faiss_index_file = "faiss_index.index"
interaction_index = faiss.read_index(faiss_index_file)

# fonction qui permet de transformer en une seule liste plusieurs liste
flatten = lambda l: [item for sublist in l for item in sublist]

def generate_next(bot_input_ids, top_k, top_p, temperature, max_length=1000, do_sample=True, pad_token=tokenizer.eos_token_id):
    # Créer un masque d'attention
    attention_mask = bot_input_ids.ne(tokenizer.pad_token_id).long()

    # Passer le masque d'attention à la fonction de génération
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

    msg = full_msg[0].tolist()
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

#===============

def query_reference_dialogs_faiss(query, index, k=5):
# Convertir la description courte en vecteur
    query_embedding = embedding_model.encode([query])   # Fonction qui convertit le texte en vecteur
    
    # Rechercher dans l'index FAISS en utilisant l'embedding
    distances, indices = index.search(query_embedding, k)
    
    # Récupérer les dialogues correspondants des champs 'Text', 'Objective' et 'Title'
    retrieved_reference_dialogs_data = []
    for idx in indices[0]:
        # Ajouter les valeurs des différents champs à la liste
        if idx < len(df):  # Vérifier que l'index est valide
            retrieved_reference_dialogs_data.append(df['Text'][idx])
            retrieved_reference_dialogs_data.append(df['Objective'][idx])
            retrieved_reference_dialogs_data.append(df['Title'][idx])

    # Utiliser un set pour éliminer les doublons, si nécessaire
    retrieved_reference_dialogs_data = list(set(retrieved_reference_dialogs_data))

    print(retrieved_reference_dialogs_data)
    return retrieved_reference_dialogs_data


# Main
if __name__ == "__main__":
    #Menu du niveau de créativité
    lvl = -1
    while lvl not in [0, 1, 2, 3]:
        lvl = int(input("What level of creativity you want to allow your npc to be?\n1 - Creative \n2 - Normal \n3 - Precise \n0 - Quit\n"))
    #Niveaux
    if lvl == 0:
        print("Exiting...")
        sys.exit()
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

    dialog_hx = []  # Initialiser l'historique des dialogues

    short_persona = input("Describe the personality of the NPC : (ex: 'greedy and suspicious merchent')\n>> ")
    name=("How do you want to name your NPC (ex: 'Lucas')\n>>")
    emotion = input("How your NPC is feeling ? (ex: 'Sad,Stressed,Happy')\n>> ") 
    context = input("What is the context of the actual situation ? (ex: 'He sells potions in a black market')\n>> ") 
    action = input("How should act or is acting the NPC ? (ex: 'he must be distrustful of new clients')\n>> ") 
    time_context = input("In which environnement does the NPC evolve (ex : 'It is night in the village, and most people are asleep.')\n")

    query = f"{short_persona} {name} {emotion} {context} {action} {time_context}"

    # ---- Ajout de la partie RAG pour interroger le dataset d'interactions ----
    retrieved_interaction_data = query_reference_dialogs_faiss(query, interaction_index)  # faiss_interaction_index est ton index FAISS des interactions
    related_dialogs = ' '.join(retrieved_interaction_data)  # Combiner les résultats en une seule chaîne de caractères
        
    combined_input = (
    f"Here is the name of the NPC you are playing : <|character_name|>{name}. "
    f"Here is the personality NPC you are playing : <|p2|>{short_persona}. "
    f"Here is how the NPC you are playing feels : <|emotion|>{emotion}. "
    f"Here is how the NPC you are playing need to act :<|action|>{action}. "
    f"Here is in what situation the NPC you are playing actually he is : <|context|>{context}. "
    f"Here is in which environnement NPC you are playing is evolving : <|time|>{time_context}. "
    )
    if related_dialogs:  
        combined_input += f"Here are some more realistic NPC dialogues from which you need to inspirate yourself based on the personality you were given: <|ref|> {related_dialogs}. Now before starting to play your role great the user neutraly and then start playing your character"


    bot_input_ids = to_var([combined_input]).long()
    # Générer la réponse avec le modèle PersonaGPT
    msg = generate_next(bot_input_ids, top_k, top_p, temperature)
    dialog_hx.append(msg)
    print("NPC : {}".format(tokenizer.decode(msg, skip_special_tokens=True)))

    while True:
        print("exit or quit to stop the conversation")
        user_inp = "<|start|>" +  input(">> User: ") + tokenizer.eos_token

        if user_inp.lower() in ['exit', 'quit']:
            print("Exiting...")
            break

        user_inp_encoded = tokenizer.encode(user_inp)
    
        # Append to the chat history
        dialog_hx.append(user_inp_encoded)

        # Limiter l'historique de la conversation à 1000 tokens
        bot_input_ids = to_var([user_inp_encoded+ flatten(dialog_hx)]).long()
    
        # Générer la réponse avec le modèle PersonaGPT
        msg = generate_next(bot_input_ids, top_k, top_p, temperature)
        dialog_hx.append(msg)
    
        # Afficher la réponse
        print("NPC : {}".format(tokenizer.decode(msg, skip_special_tokens=True)))








 #utility
#deprecated version
#flatten = lambda l: [item for sublist in l for item in sublist]

#deprecated
#if torch.cuda.is_available():
#    model = model.cuda()


#=======================

#def query_interaction_faiss(user_query, interaction_index, k=5):
    # Convertir la requête utilisateur en vecteur
#    user_query_embedding = embedding_model.encode([user_query])  # Fonction qui convertit le texte en vecteur
    # Rechercher dans l'index FAISS des interactions
#    distances, indices = interaction_index.search(user_query_embedding, k)
    
    # Récupérer les dialogues ou interactions correspondants
#
#    retrieved_interaction_data = list(set([df['Text'][idx] for idx in indices[0]]))  # Assurez-vous que retrieve_from_index est défini
#   print(retrieved_interaction_data)
#   return retrieved_interaction_data



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



#code pour utiliser un autre data set de personnalité 
# Interroger FAISS pour récupérer des traits de personnalité
    #retrieved_personality_data = query_personality_faiss(short_persona, personality_index)  # faiss_personality_index est ton index FAISS des personnalités
    #retrieved_personality = ' '.join(retrieved_personality_data)  # Combiner les résultats en une seule chaîne de caractères

    # Afficher les résultats récupérés pour information
    #print(f"Retrieved personality traits from FAISS: {retrieved_personality}")

    # Combiner la description initiale avec les résultats de FAISS pour créer une personnalité complète
    #complete_personality = short_persona + " " + retrieved_personality + tokenizer.eos_token



    #def query_reference_dialogs_faiss(query, index, k=20):
    # Convertir la description courte en vecteur
    #query_embedding = embedding_model.encode([query])   # Fonction qui convertit le texte en vecteur
    # Rechercher dans l'index FAISS des personnalités
    #distances, indices = index.search(query_embedding, k)
    
    # Récupérer les traits de personnalité correspondants
    #retrieved_reference_dialogs_data = list(set([df['Text'][idx] for idx in indices[0]])) 

    #print(retrieved_reference_dialogs_data)
    #return retrieved_reference_dialogs_data
