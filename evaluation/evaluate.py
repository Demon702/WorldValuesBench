import pandas as pd
from collections import Counter
import re
import os
import json
from scipy.stats import wasserstein_distance
import numpy as np
from typing import List, Union, Dict
import argparse

CUR_DIR = os.path.dirname(os.path.realpath(__file__))

DATASET_DIR =  os.path.join(CUR_DIR, '../WorldValuesBench')
DATASET_CONSTRUCTION_DIR = os.path.join(CUR_DIR, '../dataset_construction')

# Loading the question metadata
question_metadata = {}
with open(f'{DATASET_DIR}/question_metadata.json') as f:
    question_metadata = json.load(f)

# Loading the full qa and demographic data
full_qa_df = pd.read_csv(f'{DATASET_DIR}/full/full_value_qa.tsv', sep='\t', index_col='D_INTERVIEW')
full_demographic_df = pd.read_csv(f'{DATASET_DIR}/full/full_demographic_qa.tsv', sep='\t', index_col='D_INTERVIEW')


# Loading the codebook
codebook = {}
with open(f'{DATASET_CONSTRUCTION_DIR}/codebook.json', 'r') as f:
    codebook = json.load(f)


# Get a distribution of answers given raw answers
def get_distribution(raw_answers: List[int], question: str) -> List[float]:
    '''
    Get the distribution of answers for a question

    - raw_answers: List of raw answers
    - question: Question key

    '''
    q_info = question_metadata[question]
    min_range, max_range = q_info['answer_scale_min'], q_info['answer_scale_max']
    bin_edges = np.linspace(min_range, max_range + 1, max_range - min_range + 2) - 0.5
    hist, bin_edges = np.histogram(raw_answers, bins=bin_edges, density=True)
    return hist

def get_gt_answers(question: str, df: pd.DataFrame, normalize: bool = False) -> List[Union[int, float]]:
    '''
    Get the ground truth answers for a question

    - question: Question key
    - df: Dataframe containing the model question id, participant id and answers,
    - normalize: whether to normalize the answers

    '''
    # Extract sample interview_ids
    participant_ids = df[df['QUESTION_ID'] == question]['PARTICIPANT_ID'].tolist()

    # Extract ground truth answers
    gt_answers = full_qa_df.loc[participant_ids][question].astype(int).tolist()
    if normalize:
        gt_answers = get_normalized_scores(gt_answers, question)
    return gt_answers


def get_normalized_scores(scores: List[int], question: str, min_: Union[int, float] = 0.0, max_ : Union[int, float] = 1.0) -> List[float]:
    '''
    
    Normalize the scores between min_ and max_

    - scores: List of scores
    - question: Question key
    - min_: Minimum value for normalization
    - max_: Maximum value for normalization
    
    '''
    q_info = question_metadata[question]
    min_range, max_range = q_info['answer_scale_min'], q_info['answer_scale_max']
    scores = np.array(scores)
    normalized_scores = min_ + ((scores - min_range) * max_ / (max_range - min_range))
    return normalized_scores.tolist()

def uniform_distribution(question: str, normalize: bool = False) -> List[int]:
    '''
    Get the uniform distribution of answers for a question
    
    - question: Question key
    - normalize: whether to normalize the scores between 0-1

    '''
    q_info = question_metadata[question]
    min_range, max_range = q_info['answer_scale_min'], q_info['answer_scale_max']
    uniform_dist = list(range(min_range, max_range + 1))
    if normalize:
        uniform_dist = get_normalized_scores(uniform_dist, question)
    return uniform_dist

def majority_distribution(gt_answers: List[Union[int, float]]) -> List[Union[int, float]]:
    '''
    Get the majority distribution array from ground truth answers
    
    - gt_answers: Ground truth answers
    
    '''
    counter = Counter(gt_answers)
    most_common_element, _ = counter.most_common(1)[0]  # Get the most frequent element
    majority_dist = [most_common_element]
    return majority_dist
    

def earth_movers_distance(arr1: List[int], arr2: List[int]) -> float:
    '''
    Calculate the Earth Mover's distance or Wasserstein 1-distance (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html) between two arrays

    - arr1: Array 1
    - arr2: Array 2

    '''
    return wasserstein_distance(arr1, arr2)


def process_csv(filename: str, sep: str = ',') -> pd.DataFrame:

    '''
    Process the csv file and return the dataframe, ignore rows with invalid scores
    
    filename: Name of the file to be processed, the file should be in csv format and should have the following three columns:
    - QUESTION_ID: ID of the question
    - PARTICIPANT_ID: ID of the participant (D_INTERVIEW in the original csv)
    - SCORE: Score produced by the model

    '''

    print(f'Processing {filename}')
    df = pd.read_csv(filename, sep=sep)
    df['SCORE'] = pd.to_numeric(df['SCORE'], errors='coerce')
    
    print(f'No of valid scores', df['SCORE'].notna().sum())
    df['SCORE'] = df['SCORE'].dropna()
    return df


def evaluate(input_file: Union[str, pd.DataFrame], output_file: str = None, sep: str = ',') -> Dict[str, float]:
    '''
    Evaluate the model using the input and output files

    - input_file: Input file containing the model responses
    - output_file: Output file to write the evaluation results

    '''

    if isinstance(input_file, str):
        df = process_csv(input_file, sep=sep)
    else:
        df = input_file

    # Get the unique question ids
    questions = df['QUESTION_ID'].unique()

    output_json = {}

    for question in questions:
        gt_answers = get_gt_answers(question, df, normalize=True)

        model_scores = df[df['QUESTION_ID'] == question]['SCORE'].dropna().astype(int).tolist()
        model_scores = get_normalized_scores(model_scores, question)

        output_json[question] = earth_movers_distance(gt_answers, model_scores)

    if output_file:
        with open(output_file, 'w+') as f:
            json.dump(output_json, f, indent=4)

    return output_json

def add_demographic_col_values(df: pd.DataFrame, demographic_col: str) -> None:
    '''
    Add a column containing demohraphic attributes of a demographic column

    - df: Dataframe containing the demographic column
    - demographic_col: Demographic column name

    '''
    participant_ids = df['PARTICIPANT_ID'].tolist()
    demographic_col_values = full_demographic_df.loc[participant_ids][demographic_col].tolist()
    df[demographic_col] = demographic_col_values


def get_valid_demographic_attributes(demographic_col: str) -> List[str]:
    '''
    Get the valid unique values of a demographic column

    - demographic_col: Demographic column name

    '''
    attribute_mapping = codebook[demographic_col]['choices']
    valid_attributes = []
    for attr_code, attribute in attribute_mapping.items():

        # Negative codes correspond to negative values
        if int(attr_code) >= 0:
            valid_attributes.append(attribute)
    return valid_attributes


def get_demographic_partitions(model_to_df: Dict[str, pd.DataFrame], demographic_col: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    '''
    Get the demographic partitions for the model

    - model_to_df: Dictionary containing the model name and the corresponding dataframe
    - demographic_col: Demographic column name

    '''
    partitions = {}
    demographic_attributes = get_valid_demographic_attributes(demographic_col)

    for model_df in model_to_df.values():
        add_demographic_col_values(model_df, demographic_col) 

    for dem_attr in demographic_attributes:
        partitions[dem_attr] = {}
        for model_name, model_df in model_to_df.items():
            partitions[dem_attr][model_name] = model_df[model_df[demographic_col] == dem_attr]
    
    return partitions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, help='Input file containing the model responses', required=True)
    parser.add_argument('--input-file-separator', type=str, help='Separator for the input file', default=',')
    parser.add_argument('--output-file', type=str, help='Output file containing the evaluation results')

    args = parser.parse_args()
    evaluate(args.input_file, args.output_file, sep=args.input_file_separator)

    
    