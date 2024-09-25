import json
import pandas as pd

# Convertir le JSON en CSV
def convert_json_to_csv(json_file, csv_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convertir la liste de dictionnaires en DataFrame
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False, encoding='utf-8')

# Utilisation
convert_json_to_csv('npc-dataset.json', 'npc-dataset.csv')