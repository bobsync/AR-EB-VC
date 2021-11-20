import pandas as pd
from sklearn.svm import SVC

def posMax(array):
    tmp = 0
    for i in range(len(array)):
        if array[i] > tmp:
            tmp = array[i]
            index = i

    return index

def dynamicGraph(predict):

    i = 0
    coda = []
    sequenza = 20
    new_array = []
    counting = [0 for _ in range(0,5)]

    for _ in range(len(predict)):
        counting[predict[i]] += 1 # incremento di 1 la lista counting nella posizione del numero che leggo
        if len(coda) > sequenza:
            counting[coda[0]] -= 1
            del coda[0]
        coda.append(predict[i])
        #print(coda)
        #print(f"{elem}-{elem + sequenza}: {predict[i]} --> {posMax(counting)}")
        i += 1
        new_array.append(posMax(counting))
    
    return new_array

def getPrediction(model, FPS, id):

    # train (annotazioni) e test (senza annotazioni)
    train = pd.read_csv(f"groundtruth/sequence/{FPS}fps.csv")
    test = pd.read_csv(f"openface/{FPS}fps/{id}.csv")

    # Controllare che {id} non sia all'interno del train, e se c'è rimuoverlo
    train = train[train["id"] != id]

    train.drop(["id", "inizio", "fine"], axis = 1, inplace = True)

    # y_test non esiste perché non sono state effettuate le annotazioni
    X_train, y_train = train.drop("classe", axis = 1).copy(), train["classe"].copy()

    # Creo, alleno e vedo i risultati
    clf = model().fit(X_train, y_train)
    results = clf.predict(test)

    # Salvo i risultati in un file csv a due colonne [predict, dynamic]
    pd.DataFrame(list(zip(results, dynamicGraph(results))), columns = ["predict", "dynamic"]).to_csv(f"predictions/{id}.csv")

getPrediction(SVC, 15, 17)