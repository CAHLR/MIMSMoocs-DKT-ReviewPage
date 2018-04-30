import pandas as pd
import numpy as np
import json
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import argparse

from DKT import DKTnet
from data_helper import load_data, one_hot, preprocess

def get_callbacks():
    '''
    Some callback functions that you may find useful.
    Please refer to https://keras.io/callbacks/ for more detailed explaination
    '''
    checkpoint = ModelCheckpoint('my_model',
                                 monitor='val_loss',
                                 verbose=2,
                                 save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=2)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=0, min_lr=1e-4)
    return [checkpoint, early_stopping, reduce_lr]

def train(skill_array, response_array, skill_response_array):
    '''
    train the model with
    '''
    input_dim = skill_response_array.shape[-1] #2* num_skills
    input_dim_order = skill_array.shape[-1] #num_skills
    hidden_layer_size = 40
    batch_size = 32
    epochs = 5

    dkt = DKTnet(input_dim,
                 input_dim_order,
                 hidden_layer_size,
                 batch_size,
                 epochs,
                 x_train=skill_response_array[:, :-1, :],
                 y_train=response_array[:, :-1, np.newaxis],
                 y_train_order=skill_array[:, 1:, :],
                 validation_split=0.1,
                 validation_data=None,
                 optimizer='adam',
                 callbacks=get_callbacks())
    dkt.build()
    dkt.fit_data()


def batch_generator(skill_array, response_array, skill_response_array, batch_size=64, shuffle=True):
    """
    return: batches of data from the original data set for training
    """
    sample_num = skill_array.shape[0]
    if shuffle:
        shuffled_indices = np.random.permutation(sample_num)
        skill_array = skill_array.copy()[shuffled_indices]
        response_array = response_array.copy()[shuffled_indices]
        skill_response_array = skill_response_array.copy()[shuffled_indices]
        print('Training set shuffled')
    for ndx in range(0, sample_num, batch_size):
        skill_array_batch = skill_array[ndx:min(ndx+batch_size, sample_num)]
        response_array_batch = response_array[ndx:min(ndx+batch_size, sample_num)]
        skill_response_array_batch = skill_response_array[ndx:min(ndx+batch_size, sample_num)]
        yield skill_response_array_batch, response_array_batch, skill_array_batch

def create_validation_data(skill_array, response_array, skill_response_array,size=0.2):
    """
    return: split of data from the original data set for testing
    """
    sample_num = skill_array.shape[0]
    shuffled_indices = np.random.permutation(sample_num)
    skill_array = skill_array.copy()[shuffled_indices]
    response_array = response_array.copy()[shuffled_indices]
    skill_response_array = skill_response_array.copy()[shuffled_indices]
    print('Data set shuffled, preparing for split')
    split_index=int(size*sample_num)
    train_index=sample_num-split_index
    print('train:', train_index, "test:",split_index)
    skill_array_train = skill_array[0:train_index]
    response_array_train = response_array[0:train_index]
    skill_response_array_train = skill_response_array[0:train_index]

    skill_array_test = skill_array[train_index:]
    response_array_test = response_array[train_index:]
    skill_response_array_test = skill_response_array[train_index:]

    return skill_array_train, response_array_train, skill_response_array_train,skill_array_test,response_array_test,skill_response_array_test


def train_on_batch(skill_array, response_array, skill_response_array):
    """This function creates a DKT MODEL object using DKT.py and
    trains it using the batched data"""

    input_dim = skill_response_array.shape[-1]  #2* num_skills
    input_dim_order = skill_array.shape[-1] #num_skills
    hidden_layer_size = 40
    batch_size = 500
    epochs = 5
    print("batch size=",batch_size)
    print("epoch size=", epochs)


    '''
    parameters like batch_size, epochs x_train, y_train, y_train_order and validations are useless
    if you are doing a training by batch
    '''
    dkt = DKTnet(input_dim,
                 input_dim_order,
                 hidden_layer_size,
                 batch_size,
                 epochs,
                 x_train=skill_response_array[:, :-1, :],
                 y_train=response_array[:, :-1, np.newaxis],
                 y_train_order=skill_array[:, 1:, :],
                 validation_split=0.2,
                 validation_data=0.2,
                 optimizer='adam',
                 callbacks=None)

    dkt.build_train_on_batch()

    '''
    For simplification, we are over fitting on the training set here.
    In your model, you should do a train-test split or cross-validation which can be found in sklearn package.
    '''
    skill_array_train, response_array_train, skill_response_array_train,skill_array_test,response_array_test,skill_response_array_test=create_validation_data(skill_array, response_array, skill_response_array,size=dkt.validation_split)




    for e in range(epochs):
        print('***Epoch', e+1, 'starts****')
        iteration = 0
        total_iteration_num = 1 + (skill_array.shape[0] - 1) // batch_size
        for skill_response_array_batch, response_array_batch, skill_array_batch in batch_generator(skill_array_train, response_array_train, skill_response_array_train, batch_size=batch_size):
            dkt.train_on_batch(skill_response_array_batch[:, :-1, :],response_array_batch[:, :-1, np.newaxis],  skill_array_batch[:, 1:, :])
            iteration += 1
        print("iter: {}/{} done".format(iteration, total_iteration_num))

        result = dkt.test_on_batch(skill_response_array_test[:, :-1, :],response_array_test[:, :-1, np.newaxis],  skill_array_test[:, 1:, :])
#         print('Eval result', result)
        trainresult = dkt.test_on_batch(skill_response_array_train[:, :-1, :],response_array_train[:, :-1, np.newaxis],  skill_array_train[:, 1:, :])
        print('Evalutaion training data result', trainresult)
        print('Evalutaion validation data result', result)
        '''
        '''
        You should implement your own evaluation function here to evaluate your result on the validation set if each sample have different timesteps
        '''
    prediction = dkt.predict(skill_response_array[0:1, :-1, :],skill_array[0:1, 1:, :])

    print('Check Prediction Output:', prediction,response_array[:, :-1, np.newaxis])
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-S',"--skill_df", type=str, required=True)
    parser.add_argument('-R',"--response_df",type=str, required=True)
    parser.add_argument('-dict',"--skill_dict", type=str, required=True)
    args = parser.parse_args()
    print(args)

    response_df,skill_df,skill_dict=load_data(args.skill_df,args.response_df,args.skill_dict)
    # response_df,skill_df,skill_dict=load_data()
    skills_num = len(skill_dict)
    print('Number of skills are :{}'.format(skills_num))
    skill_array, response_array, skill_response_array = preprocess(skill_df, response_df, skills_num)
    # train(skill_array, response_array, skill_response_array)
    train_on_batch(skill_array, response_array, skill_response_array)



    # python train.py -S skill_df_delft_15.csv -R response_df_delft_15.csv -dict skill_dict_delft_all.json
    
