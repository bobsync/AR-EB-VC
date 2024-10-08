import pandas as pd
from statistics import mean, stdev
from scipy.stats import skew, kurtosis

AUS = ["AU01_r", 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']
STATS = ["min", "max", "mean", "stdev", "skew", "kurt"]

def make_csv_of_sequences(id, len_seq):

    csv = pd.read_csv(f"openface/csv/out{id}.csv") # file di output di openface

    df = csv[AUS] # prendo solamente le colonne degli action units
    df.index += 1 # faccio iniziare l'index a 1

    FEATURES = [x + "_" + y for y in AUS for x in STATS]
    COLUMNS = FEATURES

    listone = [] # nuovo dataframe

    for i in range(len(df) - len_seq):

        df_seq = df[i:i + len_seq] # sub_df della sequenza
        ls_seq = []

        for feature in range(17): # per ogni feature
            arr_seq = [df_seq.iloc[x, feature] for x in range(len(df_seq))] # sequenza della singola feature
            
            stats = [
                min(arr_seq), 
                max(arr_seq), 
                mean(arr_seq), 
                stdev(arr_seq), 
                skew(arr_seq), 
                kurtosis(arr_seq)]

            for col in [round(x, 4) for x in stats]:
                    ls_seq.append(col)

        listone.append(ls_seq)

    df_seq = pd.DataFrame(listone, columns = COLUMNS)
    df_seq.index += 1

    df_seq.to_csv(f"openface/{len_seq}fps/{id}.csv", index = False)

for FPS in [20, 25]:
    for id in [17]:
        make_csv_of_sequences(id, FPS)