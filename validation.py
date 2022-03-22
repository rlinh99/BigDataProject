from sklearn.metrics import f1_score


def show_f1_score(true, predict):
    print(f"Micro F1 Score is {f1_score(y_true=true, y_pred=predict, average='micro')}")
    print(f"Macro F1 Score is {f1_score(y_true=true, y_pred=predict, average='macro')}")

