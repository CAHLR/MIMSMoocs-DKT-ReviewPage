 


import pandas as pd
import numpy as np
import json
import os

 
""" ONE HOT ENCODING OF SKILL AND RESPONSE DATA"""  


def load_data(skill_csv,response_csv,skill_json,problem_num,is_behaviour=False):
    """
    This function reads data files:
    1. skill_dict:reads data from __skill_dict.json__ which maps skills to numbers 1,2..
    2. skill_csv: reads data from __skill.csv__ which is a sequence of exercise or skills attempted by a student
    3. response_csv: reads data from reponse.csv which is sequence of binary response to skill/exercise in skill.csv by each student
    4. problem_num: total number of problem events in the course, each tagged from 1 to problem_num in skill_json which a dictionary that maps events to integers
    5. is_behaviour is the boolean value which is TRUE when we want to include behavior events in our model or false for phase-1 dkt

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
#     print(2)
    skill_dict=json.load(open(skill_json))
#     print(3)

    print(response_df.shape,skill_df.shape)
    if is_behaviour is False:
        non_problem_event_num=0
    else:
        non_problem_event_num=len(skill_dict)-problem_num
    return skill_df,response_df,problem_num, non_problem_event_num

    
def preprocess(skill_df, response_df, problem_num,non_problem_num):
    """
    This function extracts the skills and responses from the loaded files excluding student ids
    :param skill_df: skills attempted by a student at each timestep
    :param response_df: responses on the exercises  by a student at each timestep
    :param problem_num: Total number of problem events in the course
    :param non_problem_num: Total number of non-problem events 
    """


    
    #skill_matrix = skill_df.iloc[:, 1:].values - 1 # 0 indexing skill numbers
    
    skill_matrix = skill_df.iloc[:, 1:].values
    
#     1. skill array
    skill_array =convert_event_to_one_hot(skill_matrix, problem_num+non_problem_num)
    
#     2. response array
    response_array = response_df.iloc[:, 1:].values
    
#     3. skill_response array
    skill_response_array = append_response_one_hot(skill_array,skill_matrix, response_array,problem_num)
    
    print('skill_array.shape,response_array.shape,skill_response_array.shape')
    print(skill_array.shape,response_array.shape,skill_response_array.shape)
    
    return skill_array, response_array, skill_response_array




def convert_event_to_one_hot(skill_matrix, vocab_size):
    # 1. Create one hot encode row for each integer from 0 to vocab_size
    print(vocab_size)
    one_hot_dict = np.eye(vocab_size) 
    # 2. TO ASSIGN [000] to all out of vocab_size numbers, add last row with all zeros
    one_hot_dict = np.vstack((one_hot_dict,np.zeros(vocab_size)))
#     ADDED ONE MORE ROW SO THAT  IF WE FILL NA WITH -1 then -1-1= -2 ,
# so 2nd last row becomes the assigned value
    one_hot_dict = np.vstack((one_hot_dict,np.zeros(vocab_size)))
    
    
    # sequence length is the number timesteps in the sequence
    sequence_len = skill_matrix.shape[1] 
    # instantiation of a numpy array with all elements 0
    on_hot_sequence = np.empty((skill_matrix.shape[0],sequence_len, vocab_size))

    for row in range(skill_matrix.shape[0]):
        # set vocabulary values in skills sequence of a student equal to 1, index of skills starts at 1
        on_hot_sequence[row] = one_hot_dict [skill_matrix[row]-1] #grab 1-hot rows for integers in skill_matrix
    print('Check encode in skill and one hot skill:', len(skill_matrix[skill_matrix!=0]),np.sum( on_hot_sequence))
    return on_hot_sequence




def append_response_one_hot(skill_array,skill_matrix, response_matrix, problem_num):
    """
   params:
       skill_matrix: 2-D matrix (student, skills)
       response_matrix:  2-D matrix (student, responses)
    with_response_vocab_size: Number of (2*problem events AND non-problem events) in the course
   returns:
       a 3d-darray with a shape like (student, sequence_len,problem_num)
   """
    # 1. Create one hot encode row for each integer from 0 to problem_num
    one_hot_dict_problems_only = np.eye(problem_num) 
    # 2. TO ASSIGN [000] to all out of vocab_size numbers
    one_hot_dict_problems_only= np.vstack((one_hot_dict_problems_only,np.zeros(problem_num)))
    #     ADDED ONE MORE ROW SO THAT  IF WE FILL NA WITH -1 then -1-1= -2 ,
# so 2nd last row becomes the assigned value
    one_hot_dict_problems_only= np.vstack((one_hot_dict_problems_only,np.zeros(problem_num))) 
    # 3. Get problem  ids of problems that are correct so that they can be assigned 1 value in one hot encode
#     all other  are 0, - ----this is why we encode are not encoding  any skill as  ----
    skill_matrix_only_correct_problems=skill_matrix*response_matrix

  
    # sequence length is the number timesteps in the sequence
    sequence_len = skill_matrix.shape[1] 
    # instantiation of a numpy array with all elements 0
    response_one_hot = np.empty((skill_matrix.shape[0],sequence_len, problem_num))
    # iterate over each student in data
    for i in range(response_matrix.shape[0]):
        # set vocabulary values in skills attempted by a student equal to 1
        # get encodes for sequences with non-zero problem
        response_one_hot[i]=one_hot_dict_problems_only[skill_matrix_only_correct_problems[i]-1]
        
    print('Check number correct in response and one hot response:', np.sum(response_matrix),np.sum(response_one_hot))
    skill_response_array=np.concatenate((skill_array,response_one_hot),axis=2)
    print('skill_array.shape,response_one_hot.shape,skill_response_array.shape')
    print(skill_array.shape,response_one_hot.shape,skill_response_array.shape)
    return  skill_response_array






