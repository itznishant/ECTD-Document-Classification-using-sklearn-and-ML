#!/usr/bin/python
# coding: utf-8
# ### Importing Libraries

# In[1]:
#main_file = 'Naive_Bayes_Classifer_main_training_file_test_CLEANED.py'


import sklearn
import string
import pandas as pd, numpy as np
import PyPDF2
import os, pickle
from time import time
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


# In[2]:


t0 = time()
testfiles=[]
path=str(input("Enter test directory path: \n"))
try:
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".pdf"):
                testfiles.append(os.path.join(root, file))
 # print(testfiles)            
    testfile_content = []
    for file in testfiles:
        content_data= ""
        PDF_fileObj2 = open(file, 'rb')
        pdfReader = PyPDF2.PdfFileReader(PDF_fileObj2)
        for i in range(0 , pdfReader.numPages):
            pageObj = pdfReader.getPage(i)
            if i <=3:
                content_text = pageObj.extractText()
                content_data += content_text
        testfile_content.append(content_data)
    test_data = pd.DataFrame(testfile_content)
    test_data.columns = [['TEXT']]
    test_data.info()

except Exception as e:
    print("Invalid Path!\nPlease enter correct path.")


# In[3]:

#Keywords file
Section_keywords = pd.read_excel("")  #Keywords filepath
Keywords_df = Section_keywords.copy().fillna(0)
Keywords_df = Keywords_df.drop(Keywords_df.columns[:3],axis=1)

keyword_list = []
for col in Keywords_df.columns:
    for val in Keywords_df[col]:
        if val != 0:
            keyword_list.append(val.lower())      
keyword_list_unique = set(keyword_list)


# ### Cleaning & Standardising test document

# In[4]:


for i in range(0,len(test_data['TEXT'])):
    test_data.iloc[i]['TEXT'] = test_data.iloc[i]['TEXT'].str.lower().replace("\n"," ").replace("\t"," ").str.strip(" ")
    
for i in range(len(test_data)):
    temp = " ".join([w for w in test_data.iloc[i]])
    test_data.iloc[i] = " ".join([w for w in temp.split(" ") if w in keyword_list_unique])
    
test_data


# ### load the model & vocab

# In[5]:


import pickle
model_path = "" #Model_filepath
vocab = ""      #Vocabulary_filepath
loaded_vectorizer = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open(vocab, "rb")))
model_loaded = pickle.load(open(model_path, 'rb'))


# ### Prediction

# In[6]:


predicted_labels = []
for i in range(len(test_data)):
    test_cv = loaded_vectorizer.transform(test_data.iloc[i])
    predicted_labels.append(model_loaded.predict(test_cv.toarray()))
    print(" Filename: {}\t,  Index:{}\t, Predicted Label:{}".format(testfiles[i],  i, model_loaded.predict(test_cv.toarray())))
#     print("Predicted Label: " , model_loaded.predict(test_cv.toarray()))


# ### Taking Feedback as Input
## Feedback section

get_labels = str(input("Enter Index and Correct Label seperated by comma:\n"))
get_labels_list = get_labels.split(",")
index = int(get_labels_list[0])
label = [get_labels_list[1]]


# In[8]:

predicted_labels_copy = predicted_labels
predicted_labels_copy.insert(index,label)
predicted_labels_copy.remove(predicted_labels_copy[index+1])
print([str(val) for val in predicted_labels_copy])


# ### Taking the data for label

# In[9]:


feedback_data = test_data.iloc[[index]]
feedback_data.index=[0]
# print(feedback_data)


# ### Re-Training data

# In[53]:


Retraining_data_path = ""  #Retraining data directory path
retrain_df = pd.concat([feedback_data,pd.DataFrame([label],columns=["LABEL"])],axis=1)
retrain_df.columns=[['TEXT','LABEL']]
print(retrain_df)
retrain_df.to_excel(os.path.join(Retraining_data_path,"Retrain_data_" +str(str(time()).split(".")[0]) + ".xlsx"))


# ### Re-Training the Model

# In[63]:


import glob
retrain_data = [pd.read_excel(file,index_col=[0]) for file in glob.glob("")]  #Collecting all training data from excel files
full_retrain_data = pd.concat(retrain_data, ignore_index=1)
full_retrain_data.dropna(inplace=True)


# In[ ]:


text_data_df = pd.read_excel("",index_col=[0]) #Training data filepath


# In[67]:


full_data=pd.concat([text_data_df,full_retrain_data])
full_data.to_excel(os.path.join(""))  #Main Training + Feedback data filepath
print(full_data)

os.system("python """) #Specify Main code filepath
print("Time Taken: %0.3fs" % (time() - t0)) #Timer