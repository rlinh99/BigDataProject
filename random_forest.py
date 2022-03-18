from data_loader import load_data


def run_random_forest():
    raw_train_data = load_data('train')
    features = ['drugName', 'condition', 'usefulCount', 'sideEffects']
    label = 'rating'

    # handle missing condition
    for f in raw_train_data:
        

    train = raw_train_data.loc[raw_train_data['sideEffects'].isnull(), 'sideEffects']
    print(len(train))

