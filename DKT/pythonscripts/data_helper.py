import pandas as pd
import numpy as np
import json
import os


def load_data(skill_csv,response_csv,skill_json):
    """
    This function reads data files and returns:
    1. skill_dict:reads data from __skill_dict.json__ which maps skills to numbers 1,2..
    2. skill_df: reads data from __skill.tsv__ which is a sequence of exercise or skills attempted by a student
    3. response_df: reads data from correct.tsv which is sequence of binary response to skill/exercise in skill.tsv by each student
    4. assistment_df: reads data from assistment_id.tsv which is a sequence of event id of the skill/exercises

    """
    #response_df = pd.read_csv('correct.tsv', sep='\t').drop('Unnamed: 0', axis=1)
    # skill_df = pd.read_csv('skill.tsv', sep='\t').drop('Unnamed: 0', axis=1)
    # assistment_df = pd.read_csv('assistment_id.tsv', sep='\t').drop('Unnamed: 0', axis=1)
    # data_dir="data/dktData/"
    data_dir=""
    response_csv = os.path.join(data_dir,response_csv)

    print(response_csv)

    # response_df = pd.read_csv(response_csv, sep='\t').drop('Unnamed: 0', axis=1)
    response_df = pd.read_csv(response_csv)
    print(1)
    skill_csv = os.path.join(data_dir,skill_csv)
    # skill_df=pd.read_csv(skill_csv, sep='\t').drop('Unnamed: 0', axis=1)
    skill_df=pd.read_csv(skill_csv)
    print(2)
    skill_dict=json.load(open(skill_json))
    print(3)

    print(response_df.shape,skill_df.shape)
    return response_df, skill_df, skill_dict

# def load_data():
#     """
#     This function reads data files and returns:
#     1. skill_dict:reads data from __skill_dict.json__ which maps skills to numbers 1,2..
#     2. skill_df: reads data from __skill.tsv__ which is a sequence of exercise or skills attempted by a student
#     3. response_df: reads data from correct.tsv which is sequence of binary response to skill/exercise in skill.tsv by each student
#     4. assistment_df: reads data from assistment_id.tsv which is a sequence of event id of the skill/exercises

#     """
#     response_df = pd.read_csv('correct.tsv', sep='\t').drop('Unnamed: 0', axis=1)
#     skill_df = pd.read_csv('skill.tsv', sep='\t').drop('Unnamed: 0', axis=1)
#     assistment_df = pd.read_csv('assistment_id.tsv', sep='\t').drop('Unnamed: 0', axis=1)
    
#     skill_dict = {}
#     with open('skill_dict.json', 'r', encoding='utf-8') as f:
#         """This code block aims to start skill indexing from 0, 
#         which otherwise strats from 1 in skill_dict.json"""
#         loaded = json.load(f)
#         for k, v in loaded.items():
#             skill_dict[k] = int(v) - 1 # 0 indexing
#     print(response_df.shape,skill_df.shape)
#     return response_df, skill_df, assistment_df, skill_dict





def preprocess(skill_df, response_df, skill_num):
    """
    This function extracts the skills and responses from the loaded files excluding student ids
    :param skill_df: skills attempted by a student at each timestep
    :param response_df: responses on the exercises  by a student at each timestep
    :param skill_num: Total number of skills in the course
    """
    skill_matrix = skill_df.iloc[:, 1:].values - 1 # 0 indexing skill numbers
    response_array = response_df.iloc[:, 1:].values
    skill_array = one_hot(skill_matrix, skill_num)
    skill_response_array = dkt_one_hot(skill_matrix, response_array, skill_num)
    return skill_array, response_array, skill_response_array

def one_hot(skill_matrix, vocab_size):
    """
    params:
        skill_matrix: 2-D matrix (student, skills)
        vocal_size: Number of skills in the course
    returns:
        a 3d-darray with a shape like (student, sequence_len, vocab_size)
    """

    seq_len = skill_matrix.shape[1] # sequence length is the number of the skills
    # instantiation of a numpy array with all elements 0
    result = np.zeros((skill_matrix.shape[0], seq_len, vocab_size))
    # iterate over each student in data
    for i in range(skill_matrix.shape[0]):
        # set vocabulary values in skills attempted by a student equal to 1
        result[i, np.arange(seq_len), skill_matrix[i]] = 1.
    return result

def dkt_one_hot(skill_matrix, response_matrix, vocab_size):
    """
   params:
       skill_matrix: 2-D matrix (student, skills)
       response_matrix:  2-D matrix (student, responses)
       vocal_size: Number of skills in the course
   returns:
       a 3d-darray with a shape like (student, sequence_len, 2*vocab_size)
   """

    seq_len = skill_matrix.shape[1]# sequence length is the number of the skills
    # instantiation of a numpy array with all elements 0
    skill_response_array = np.zeros((skill_matrix.shape[0], seq_len, 2 * vocab_size))
    # iterate over each student in data
    for i in range(skill_matrix.shape[0]):
        # set vocabulary values in correct response attempted by a student equal to 1
        # 2* skill_matrix[i] goes to index in one hot encode for responses

        skill_response_array[i, np.arange(seq_len), 2 * skill_matrix[i] + response_matrix[i]] = 1.
    return skill_response_array



# Just for debugging:
# if __name__ == '__main__':
#     response_df, skill_df, assistment_df, skill_dict = load_data()
#     skills_num = len(skill_dict)
#     skill_array, response_array, skill_response_array = preprocess(skill_df, response_df, skills_num)
#     print(skill_df.iloc[1],skill_array[1])

