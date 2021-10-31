import pandas as pd
import json, random

def subsample_and_to_bin(json_input, num_samples=10000):
    """ 
    Uniformally samples from json_input and converts 'overall' 
    label to binary.  Values 1 and 2 are converted to '0',
    while values 4 and 5 are converted to '1'; data with value
    3 (neutral reviews) are removed from consideration.
    Saves the preprocessed dataset to a csv file.
    
    Args:
        json_input: file path of dataset, in the format of
          data/*.json
        num_samples: size of subsample. Default is 10000.
    """
    filtered_data = []

    with open(json_input) as f:
        for json_obj in f:
            d = json.loads(json_obj)
            if d['overall'] != 3: # Filter out neutral reviews
                d['overall'] = 0 if d['overall'] < 3 else 1
                filtered_data.append(d)

    random.seed(10)
    subsample = random.sample(filtered_data, num_samples)

    pd_in = {'review': [d['reviewText'] for d in subsample],
             'label':  [d['overall'] for d in subsample]}
    df = pd.DataFrame.from_dict(pd_in)

    out_file = json_input[:-5] + '_preprocessed.csv'
    df.to_csv(out_file, index=False)


def main():
    subsample_and_to_bin('data/reviews_Baby_5.json')
    d = pd.read_csv('data/reviews_Baby_5_preprocessed.csv')
    print(d.head())

if __name__ == "__main__":
    main()
