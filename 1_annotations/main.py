# File locali
from globals import printProgressBar, flatten, ACTION_COL
from frame import all_annotation_to_groundtruth
from sequence import make_csv_of_sequences

# Librerie esterne
import pandas as pd

# Variabili globali
ATTRIBUTI = ["id", "classe", "frame", "end_seq"] + ACTION_COL

print("\n# - # UPDATE CSV ANNOTAZIONI # - #\n")

# Creo il dataframe e lo salvo in formato csv (all.csv)
dataset = pd.DataFrame(flatten(all_annotation_to_groundtruth(ACTION_COL)), columns = ATTRIBUTI)
dataset.to_csv(f"groundtruth/frame/all.csv", index = False)

range_FPS = range(15, 26, 5)

print(f"\nGenerando i file csv [sequence] dai file csv [frame]: \n")

# Creo altri csv che tengono conto delle sequenze e non dei frame
for i, FPS in enumerate(range_FPS):
    DF = make_csv_of_sequences(FPS, "groundtruth/frame/all.csv")
    DF.to_csv(f"groundtruth/sequence/{FPS}fps.csv", index = False)

print()