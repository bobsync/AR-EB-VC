import subprocess, os, shutil
from globals import printProgressBar
import pandas as pd

# Descrizione delle classi
codifica = {    
    0 : "bocca_chiusa",
    1 : "parla",
    2 : "forchetta",
    3 : "mastica",
    4 : "sorride"}

def predicted_video(id, column):
    """Funzione che genera il video delle classi predette che si trovano in csv_file"""

    prediction = pd.read_csv(f"predictions/{id}.csv")
    start, tmp = 0, 0
    diz = {x:[] for x in range(5)}

    for i, elem in enumerate(prediction[column]):
        if i == 0: continue # salta la prima riga del csv (quella degli attributi)
        if tmp == 0:
            start = i
            tmp = elem
        if tmp != elem:
            diz[tmp].append((start, i))
            tmp = 0

    # Progress bar
    i, length = 0, sum([len(diz[x]) for x in diz])
    printProgressBar(i, length)

    # Creo le cartelle per i video singoli che verranno poi eliminati
    dir_name = f"video/{id}/singoli"
    shutil.rmtree(dir_name, ignore_errors = True)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)   

    # Creo i singoli video dalle annotazioni
    for classe in diz:
        for coppia in diz[classe]:
            start, end = coppia[0], coppia[1]
            bashCommand = f"ffmpeg -y -hide_banner -loglevel error -start_number {start} -i openface/aligned/out{id}_aligned/frame_det_00_%06d.bmp -frames:v {end - start} -vf pad=width=480:height=360:x=179:y=119:color=black,drawtext=fontfile=Arial.ttf:text={codifica[classe]}:x=(w-text_w)/2:y=h-th-80:fontcolor=white:fontsize=15:box=1:boxcolor=black:boxborderw=5,drawtext=fontfile=Arial.ttf:text=%{{frame_num}}:start_number={start}:x=w-tw-20:y=20:fontcolor=white:fontsize=15:box=1:boxcolor=black:boxborderw=5 -c:v libx264 -pix_fmt yuv420p {dir_name}/{start}_{end}.mp4"
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            i += 1
            printProgressBar(i, length)

    # Mi salvo la lista ordinata dei video per poi concatenarli
    with open(f"video/{id}/list.txt", "w") as f:
        arr = os.listdir(dir_name)
        if arr != []:
            # Get list of all files in a given directory sorted by name
            for el in sorted(arr, key = lambda x : int(x.split("_")[0])):
                f.write(f"file singoli/{el}\n")

    print(f"\n# - # Concatenazione video {id} - {column} in corso # - #\n")

    # Concateno i video singoli tra loro
    if arr != []:
        bashConcat = f"ffmpeg -y -hide_banner -loglevel error -f concat -safe 0 -i video/{id}/list.txt -vf drawtext=fontfile=Arial.ttf:text=ID_{id}:x=20:y=20:fontcolor=white:fontsize=15:box=1:boxcolor=black:boxborderw=5 -c:v libx264 -pix_fmt yuv420p video/{id}/{column}.mp4"
        process = subprocess.Popen(bashConcat.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

    # Elimino la cartella con i video singoli
    shutil.rmtree(dir_name, ignore_errors=True)  

print()
for column in ["predict", "dynamic"]:
    for id in [17]:
        predicted_video(id, column)