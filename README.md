# 1.FinalProjectMOOCsAdaptivity 
This repo contains the following code files: 
1. DKT   --- DKT/working_notebooks
2. Review page html format --- /Review_d3
3. Moocs Behavioural model --- Commented code


##### DKT:  DKT has these dkt modelling and preprocessing files:

dkt phase 1: dkt based only on the quizzes   
dkt phase 2 : dkt based on quizzes and behaviour 

Each phase needs three files :
1. skill_dict json
2. event_stream csv 
3. response_stream csv

Preprocess code :  
Converts edx data to a specific format as read in the modelling  files. Each student is represented by a stream of events, each encoded by a mapping from skill_dict json .
Required format:
<p align="center">
  <img src="/pic/data_dkt.png" width="800"/>
</p>



To add comments we can do any of these: 
### 1. Write comments on github online.
      Open the file.  
      Click edit (pen like symbol), make editions.   
      Commit with a descrptive message.
### 2. Write comments in local repo.
      1. Git pull the repo before you start commenting or editing.   
      2. Edit/comment in files.
      
      # to check what files have been modified
      # git status
      
      3. git add <the changed filename> 
      4. git commit -m ' a descrptive message'
      5. git push <remotename> <branchname>
      
      # e.g git push origin master
      # to see all your remotes use following command
      # git remote -v
    
