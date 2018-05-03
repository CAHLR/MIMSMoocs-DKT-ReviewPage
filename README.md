### This repo contains the following code files: 

1. DKT   --- DKT
2. Review page html format --- Review_d3
3. Moocs Behavioural model --- Commented behaviour model code


### DKT:  DKT has these dkt modelling and preprocessing files:

#### A. Preprocess code :

Converts edx data to a specific format as read in the modelling  files. Each student is represented by a stream of events, each encoded by a mapping from skill_dict json .
Required format:
<p align="center">
  <img src="dkt/pic/data_dkt.png" width="800"/>
</p>


#### B. Modelling:

dkt phase 1: dkt based only on the quizzes   
dkt phase 2 : dkt based on quizzes and behaviour  
dkt baseline: dkt based on majority class

###### Each phase needs three files :
1. skill_dictionary json
2. event_stream csv 
3. response_stream csv



### To train DKT phase 1 model:
Files in working notebooks folder, uses tensorflow backend in keras.
1. Read edx  raw log data file in __initial_preindex_from_log.ipynb__  to get a __pre_index_data.csv__ file which will have followings fields for each log entry:  
*user, timestamp, is_problem, is_correct,unique_represenation_of_event, time_spent*

2. Read the __pre_index_data.csv__ in the __phase1-create-skill-response.ipynb__ and preprocess it to get a *tagged_event_stream_per_student* data in csv format and *response_stream_per_student* in csv format. Here you can change the number of timestamps you want in your event streams.

3. Read the event_stream and response stream data from step 2 along with the skill-tag mapping  dictionary file in the __DKT-phase1-Batched.ipynb__, set the hyperparameters, they are currently hardcoded in the python notebook. Run the entire notebook to train a DKT model on the data.

Similar procedure can be used for running DKT Phase 2, using relevant files.


    


