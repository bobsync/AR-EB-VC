import pandas as pd
from statistics import mean, stdev
from scipy.stats import skew, kurtosis
from globals import printProgressBar

BASIC = ["id", "classe", "inizio", "fine"]
STATS = ["min", "max", "mean", "stdev", "skew", "kurt"]

# len_seq: numero della lunghezza della sequenza di frame
# csv: nome del file csv da leggere in input

def make_csv_of_sequences(len_seq, file_name_csv):

    csv = pd.read_csv(file_name_csv)
    df = csv.values

    FEATURES = [x + "_" + y for y in csv.columns[4:] for x in STATS]
    COLUMNS = BASIC + FEATURES # colonne del nuovo csv

    listone = [] # nuovo dataframe

    # per la print progress bar
    total = len(df)
    printProgressBar(0, total, prefix = f"{len_seq}fps")

    for i, row in enumerate(df):

        id      = int(row[0])
        start   = int(row[2])
        end     = int(row[3])

        if end - start >= len_seq:
            classe = int(df[i][1]) # classe della sequenza
            df_seq = df[i:i + len_seq] # sub_df della sequenza

            if classe == 5: continue # skippo classe 5

            ls_seq = [id, classe, start, start + len_seq - 1]

            for feature in range(4,21): # per ogni feature
                arr_seq = [df_seq[x][feature] for x in range(len(df_seq))] # sequenza della singola feature

                stats = [
                    min(arr_seq), 
                    max(arr_seq), 
                    mean(arr_seq), 
                    stdev(arr_seq), 
                    skew(arr_seq), 
                    kurtosis(arr_seq)
                ]

                stats = [round(x, 4) for x in stats]
                
                for col in stats:
                    ls_seq.append(col)

            listone.append(ls_seq)

        printProgressBar(i+1, total, prefix = f"{len_seq}fps")

    return pd.DataFrame(listone, columns = COLUMNS) 