"""
From https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp,
download "WVS Cross-National Inverted Wave 7 csv v5 0.zip" and unzip it to "WVS_Cross-National_Wave_7_csv_v5_0.csv".

This script takes in
 1. question_metadata.json -> 46 demographic questions + 240 value questions in our dataset + their metadata
 2. WVS_Cross-National_Wave_7_csv_v5_0.csv -> raw participant answers
 3. answer_adjustment.json -> in case the original answer indexes are not ordered semantically etc.
 4. codebook.json -> mapping from numerical answers in the csv to text
and outputs the dataset with full/train/valid/test splits.
"""


import pandas as pd
import json
from sklearn.model_selection import train_test_split
import random
import os
import argparse
from typing import Dict, Any

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(CUR_DIR, "../WorldValuesBench")


########## ARGUMENT PARSING ##########
parser = argparse.ArgumentParser(description='Prepare the dataset for training')
parser.add_argument('--raw-dataset-path', type=str, help='Path to raw WVS csv data downloaded from https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp')
args = parser.parse_args()


########## PREPROCESS ##########

# Convert numbers like '1.0' to '1'
def convert_to_int(val):
    """
    Convert numbers like '1.0' to '1'
    Args:
        val (str): raw answer
    Returns:
        ans (str): converted answer
    """
    try:
        int(val)
        return str(int(val))
    except:
        pass

    try:
        float(val)
        if float(val) == int(float(val)):
            return str(int(float(val)))
        return val
    except:
        return val

def preprocess(value: str, question_key: str, preprocess_map: Dict[str, Dict[str, str]]) -> str:
    """
    Takes the raw answer makes it ordinal if required
    value: raw answer
    question_key: question key
    """
    value = convert_to_int(value)
    if question_key not in preprocess_map or str(value) not in preprocess_map[question_key]:
        return value
    
    ret_val = preprocess_map[question_key][str(value)]
    return int(ret_val) if ret_val else ''
    
def process(
        value: str,
        question_key: str,
        question_metadata: Dict[str, Any],
        codebook: Dict[str, Dict[str, str]]
    ) -> str:
    """
    Process the answer and convert numeric answers to text answers for objective questions
    value: raw answer
    question_key: question key
    codebook: mapping from numerical answers in the csv to text
    """
    ques_info = question_metadata[question_key]
    if value == '':
        return ''
    if ques_info['answer_data_type'] == 'non_ordinal':
        if str(value) in codebook[question_key]['choices']:
            return codebook[question_key]['choices'][str(value)]
        return value
    elif ques_info['answer_data_type'] == 'ordinal':
        value = int(value)
        if value < 0:
            return ''
        min_range = ques_info['answer_scale_min']
        max_range = ques_info['answer_scale_max']
        if value < min_range or value > max_range:
            print('question_key', question_key)
            print('value', value)
            print('Incorrect occurred, incorrect implementation')
        return value

def main():
    ########## LOAD DATA ##########

    # Read the csv data downloaded from https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp
    df = pd.read_csv(args.raw_dataset_path)

    # filter duplicated INTERVIEW_ID rows
    # We noticed this user id appearing multiple times, so in our dataset we excluded this id
    df = df.loc[df['D_INTERVIEW'] != 858069901]


    ################# LOAD METADATA #################

    # Codebook contains the mapping from integer answer choice to the actual natural language answer
    codebook = {}
    with open(f'{CUR_DIR}/codebook.json', 'r') as f:
        codebook = json.load(f)

    # Load question metadata
    question_metadata = {}
    with open(f'{CUR_DIR}/question_metadata.json', 'r') as f:
        question_metadata = json.load(f)

    # data transformation
    # filter required questions
    req_question_keys = question_metadata.keys()
    df = df[req_question_keys]

    # get preprocessing mappings
    preprocess_map = {}

    # All answers were not initially in ordinal sequence, to make the answers ordinal we had to remap some of the answer choices
    # These mappings are stored in answer_mapping.json
    with open(f'{CUR_DIR}/answer_adjustment.json', 'r') as f:
        preprocess_map = json.load(f)

    # Preprocess all the answers
    for question_key in req_question_keys:
        df[question_key] = df[question_key].map(lambda val: preprocess(val, question_key, preprocess_map))

    # Process all the answers
    for question_key in req_question_keys:
        df[question_key] = df[question_key].map(lambda val: process(val, question_key, question_metadata, codebook))
        
    # Persona columns are the columns which contain the demographic information of the user
    demographic_columns = [q for q in req_question_keys if question_metadata[q]['use_case'] == 'demographic']
    demographic_columns.remove('D_INTERVIEW')
    demographic_columns = ['D_INTERVIEW'] + demographic_columns
    persona_df = df[demographic_columns]
    persona_df.set_index('D_INTERVIEW')

    # value columns contain the user's answer to the questions they were asked
    value_columns = ['D_INTERVIEW'] + [q for q in req_question_keys if question_metadata[q]['use_case'] == 'value']
    value_df = df[value_columns]
    value_df.set_index('D_INTERVIEW')


    ########## SPLIT ##########
    # Split the data into training, validation, and test sets
    train_ratio = 0.7  # 70% of the data for training
    valid_ratio = 0.15  # 15% of the data for validation
    test_ratio = 0.15  # 15% of the data for testing
    dataset_indices = list(value_df.index)
    random.seed(0)

    total_samples = len(df)
    random.shuffle(dataset_indices)
    train_end = int(train_ratio * total_samples)
    test_end = train_end + int(test_ratio * total_samples)

    train_indices = dataset_indices[:train_end]
    test_indices = dataset_indices[train_end:test_end]
    valid_indices = dataset_indices[test_end:]


    ########## SAVE ##########
    # The whole dataset (train + valid + test) into a single file
    os.makedirs(f'{DATASET_DIR}/full', exist_ok=True)
    persona_df.to_csv(f'{DATASET_DIR}/full/full_demographic_qa.tsv', sep='\t', index = False)
    value_df.to_csv(f'{DATASET_DIR}/full/full_value_qa.tsv', sep='\t', index = False)

    # Prepare training split of the data
    os.makedirs(f'{DATASET_DIR}/train', exist_ok=True)
    user_metadata_train = persona_df.loc[train_indices]
    user_metadata_train.to_csv(f'{DATASET_DIR}/train/train_demographic_qa.tsv', sep='\t', index=False)
    value_train = value_df.loc[train_indices]
    value_train.to_csv(f'{DATASET_DIR}/train/train_value_qa.tsv', sep='\t', index=False)

    # Prepare valid split of the data
    os.makedirs(f'{DATASET_DIR}/valid', exist_ok=True)
    user_metadata_valid = persona_df.loc[valid_indices]
    user_metadata_valid.to_csv(f'{DATASET_DIR}/valid/valid_demographic_qa.tsv', sep='\t', index=False)
    value_valid = value_df.loc[valid_indices]
    value_valid.to_csv(f'{DATASET_DIR}/valid/valid_value_qa.tsv', sep='\t', index=False)

    # Prepare test split of the data
    os.makedirs(f'{DATASET_DIR}/test', exist_ok=True)
    user_metadata_test = persona_df.loc[test_indices]
    user_metadata_test.to_csv(f'{DATASET_DIR}/test/test_demographic_qa.tsv', sep='\t', index=False)
    value_test = value_df.loc[test_indices]
    value_test.to_csv(f'{DATASET_DIR}/test/test_value_qa.tsv', sep='\t', index=False)

    # Save the question metadata
    with open(f'{DATASET_DIR}/question_metadata.json', 'w+') as f:
        json.dump(question_metadata, f, indent = 4)



if __name__ == '__main__':
    main()