import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import f1_score
from matplotlib.pyplot import savefig
from statistics import mean
from my_functions import *

def reports_to_txt(files_list, models_list, toScale = False):

    for file_name in files_list:

        df = pd.read_csv(f"groundtruth/sequence/{file_name}.csv")
        ids_list = df["id"].unique()
        print("\n" + file_name)

        for model in models_list:

            # model name to pretty print
            model_name = str(model).split(".")[-1][:-2]
            print("--- " + model_name)

            with open(f'reports/leave_one_out/{file_name}_{model_name}_{str(toScale)}_scale.txt', 'w') as f:

                f1_list = []

                for out in ids_list:

                    print(f"------- {out}")

                    # Divido in train (all subjects - 1) e test (1 subject)
                    X_train, X_test, y_train, y_test = train_test_one_subject_out(df, out, isScale = toScale)

                    # Creo e alleno il modello
                    clf = model()
                    clf.fit(X_train, y_train)
                    
                    # Predict sul modello allenato
                    p_test = clf.predict(X_test)

                    # Report sul test
                    micro_f1_test = f1_score(y_test, p_test, average = "micro")
                    macro_f1_test = f1_score(y_test, p_test, average = "macro")
                    f1_list.append((micro_f1_test, macro_f1_test))

                    # Salvo i risultati in un file txt
                    if int(out) // 10 == 0: out = f"0{out}"
                    line = f"Testing [{out}]: micro: {myRound(micro_f1_test)} macro: {myRound(macro_f1_test)}\n"
                    f.write(line)

                    # Confusion Matrix
                    plot_confusion_matrix(clf, X_test, y_test, normalize = "true")
                    savefig(f"confusion_matrix/{file_name}/ID{out}_{model_name}_{str(toScale)}_scale.png")

                mean_micro, mean_macro = mean([x[0] for x in f1_list]), mean([x[1] for x in f1_list])
                f.write(f"\nMean:         micro: {myRound(mean_micro)} macro: {myRound(mean_macro)}")

for bool in [True]:
    reports_to_txt( ["15fps", "20fps", "25fps"],
                    [SVC, RandomForestClassifier],
                    toScale=bool)