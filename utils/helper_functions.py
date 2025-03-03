import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
import json


def human_score(f,df,human_eval, corr=matthews_corrcoef):

    results = []

    # Every dataset has 120 datapoints so we try to collect the results per dataset 
    for i in range(0,600,120):
        d_results = []
    
        # Every dataset uses 5 models and so we use human_eval.columns to get the 5 model names
        for j in human_eval.columns:
            x = df[i:i+120].apply(lambda row: f(row[j], row['answers']), axis=1)
            x = corr(np.where(x > 0.5, 1,0),human_eval[i:i+120][j])
            type(x)
            d_results.append(x)
        results.append(d_results)
  

    results = np.array(results)

    return results.T


def read_custom_json(file_path):
    # Read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(data)
    
    return df


def read_evouna(file):
    # Usage
    file_path = f'/home/sai/Programs/UIC/594/Experiments_v2/other_data/QA-Eval-main/EVOUNA/{file}.json'
    evouna = read_custom_json(file_path)
    evouna = evouna[evouna['improper'] == False]
    return evouna