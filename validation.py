from sklearn.metrics import f1_score


def get_f1_score(true, predict):

    return f1_score(true, predict, average='micro'),\
           f1_score(true, predict, average='macro')