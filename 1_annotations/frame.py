import os
import pandas as pd
from globals import printProgressBar

# Crea il dataframe con le annotazioni di un partecipante [frame]
def single_annotation_to_groundtruth(id, features):

    data = []

    # Leggo il csv delle annotazioni (noted) e quello di OpenFace (out)
    try:
        ann = pd.read_csv(f"noted/{id}.csv")
        out = pd.read_csv(f"openface/csv/out{id}.csv")
        X = out[out["success"] == 1]
        X = X[features]

        # per la print progress bar
        total = len(ann.values)
        pref_id = f"0{id}" if int(id) // 10 == 0 else str(id)
        printProgressBar(0, total, prefix = pref_id)

        for i, row in enumerate(ann.values):
            classe = int(row[1])

            start = row[2]
            end = row[3]

            # skippo la classe 5
            if classe == 5: continue

            for index in range(start, end + 1):

                try:
                    out_data = X.loc[[index-1]]
                    ls = [id, classe, index, end] + out_data.values.flatten().tolist()
                    data.append(ls)

                except Exception: pass # NON TROVA QUELLI CON SUCCESS != 1

            printProgressBar(i+1, total, prefix = pref_id)

        # Colonne del file csv
        ATTRIBUTI = ["id", "classe", "frame", "end_seq"] + features
                    
        dataset = pd.DataFrame(data, columns = ATTRIBUTI)
        dataset.to_csv(f"groundtruth/frame/{id}.csv", index = False)

    except Exception: pass # NON TROVA I CSV DELLE ANNOTAZIONI

    return data

# Crea il dataframe con le annotazioni di tutti i partecipanti [frame]
def all_annotation_to_groundtruth(features, path = "noted/"):

    data = []

    # Prendo gli id dei partecipanti annotati in ordine
    noted = sorted([x.split(".")[0] for x in os.listdir(path)], key = lambda x : int(x))

    print(f"Partecipanti annotati: {[int(x) for x in noted]}")
    print(f"Generando i file csv [frame] dalle annotazioni: \n")

    # Aggiungo le annotazioni in un'unica lista
    for i, id in enumerate(noted):
        id_data = single_annotation_to_groundtruth(id, features) 

        if id_data != []:
            data.append(id_data)

    return data