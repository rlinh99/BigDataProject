import NBC, KNN
import pandas as pd



def fill_csv(result):
    df = pd.read_csv('data/testing.csv')
    df['rating'] = result
    df.to_csv('filled/filled_' + 'testing.csv')


if __name__ == '__main__':
    result = NBC.run()
    # result = KNN.run()
    fill_csv(result)
