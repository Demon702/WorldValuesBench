import pandas as pd
import numpy as np
import random
import os
import json

# Set the random seed for reproducibility
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
random.seed(0)

DATASET_DIR = os.path.join(CUR_DIR, '../../WorldValuesBench')
VALID_DATASET_DIR = f"{DATASET_DIR}/valid"
PROBE_DATASET_DIR = f"{DATASET_DIR}/probe"
DATASET_CONSTRUCTION_DIR = os.path.join(CUR_DIR, '../')

important_question_keys, important_demographic_keys = [], []

################## LOAD THE CASE STUDY QUESTIONS ################
with open(f'{PROBE_DATASET_DIR}/value_questions.json', 'r') as f:
    case_study_questions = json.load(f).keys()
    

############### LOAD QUESTION METADATA #########################
question_metadata = {}
with open(os.path.join(DATASET_CONSTRUCTION_DIR, 'question_metadata.json'), 'r') as f:
    question_metadata = json.load(f)

################## LOAD THE VALID SET ##########################
# We will use the validation set to create the probe set

valid_qa_df = pd.read_csv(os.path.join(VALID_DATASET_DIR, 'valid_value_qa.tsv'), sep='\t', index_col='D_INTERVIEW')
valid_demographic_df = pd.read_csv(os.path.join(VALID_DATASET_DIR, 'valid_demographic_qa.tsv'), sep='\t', index_col='D_INTERVIEW')

################ LOAD CODEBOOK  ################
    
codebook = {}
with open(f'{DATASET_CONSTRUCTION_DIR}/codebook.json', 'r') as f:
    codebook = json.load(f)

################ GROUPING DEMOGRAPHIC ATTRIBUTES ####################
 
# In the demography df adding a column for contitent to which the user belongs
country_continent_df = pd.read_csv(f'{PROBE_DATASET_DIR}/country2continent.csv')
country_continent_mapping = {}
for idx, row in country_continent_df.iterrows():
    country_continent_mapping[row['Country']] = row['Continent']
valid_demographic_df['Continent'] = valid_demographic_df['B_COUNTRY'].map(lambda country: country_continent_mapping[country])

# In the demography df adding a column for education level of the user
education_level_df = pd.read_csv(f'{PROBE_DATASET_DIR}/education2level.csv')

education_level_mapping = {}
for idx, row in education_level_df.iterrows():
    education_level_mapping[row['Education']] = row['Level']
valid_demographic_df['Education Level'] = valid_demographic_df['Q275'].map(lambda edu: education_level_mapping.get(edu, ''))

################ JOIN THE DEMOGRAPHIC AND QA DATAFRAMES ################

valid_df = valid_demographic_df.join(valid_qa_df, how='left')


################# EXTRACT DEMOGRAPHIC ATTRIBUTES ######################

# We study continent, urban/rural settlement type and education, so we extract the values of these attributes presnet in the dataset
# In our dataset, H_URBRURAL contains urban / rural information and Q275 contains education information of the user


all_continents = list(valid_demographic_df['Continent'].unique())
urban_rural_types = ['Urban', 'Rural']

all_education_levels = list(valid_demographic_df['Education Level'].unique())
all_education_levels = [level for level in all_education_levels if level != '']


#################  SAMPLE FOR EACH DEMOGRAPHIC COMBINATION ######################
sep = '\t'
with open(f'{PROBE_DATASET_DIR}/samples.tsv', 'w+') as samples_file:
    sample_file_columns = ['Question', 'Question Category', 'Continent', 'Urban / Rural', 'Education', 'D_INTERVIEW']
    samples_file.write(sep.join(sample_file_columns))
    samples_file.write('\n')

    
    for q_key in case_study_questions:
        question_category = question_metadata[q_key]['category']
        no_of_partitions = 0
        filtered_df = valid_df[valid_df[q_key].notnull()]
        for continent in all_continents:
            for urban_rural in urban_rural_types:
                for education_level in all_education_levels:
                    partition_df = filtered_df[(filtered_df['Continent'] == continent) & (filtered_df['H_URBRURAL'] == urban_rural) & (filtered_df['Education Level'] == education_level)]
                    filtered_indexes = list(partition_df.index)
                    if len(filtered_indexes) <= threshold:
                        print('question', q_key, 'continent', continent, 'urban_rural', urban_rural, 'education_level', education_level, 'has less than 5 samples')
                        continue
                    no_of_partitions += 1
                    choices = random.sample(filtered_indexes, k=min(samples_per_partition, len(filtered_indexes)))
                    for c in choices:
                        samples_file.write(sep.join([q_key, question_category, continent, urban_rural, education_level, str(c)]))
                        samples_file.write('\n')
        print('question', q_key, 'no of partitions', no_of_partitions)
        


