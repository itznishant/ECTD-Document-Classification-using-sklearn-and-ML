#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn
import string, re
import pandas as pd, numpy as np
import PyPDF2
import os, pickle
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics     import accuracy_score,confusion_matrix,f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from collections import defaultdict
# from sklearn.model_selection import KFold, cross_val_score


# In[3]:


# files_list = []
# for root, dirs, files in os.walk(""):    #PDF documents filepath
#     for file in files:
#         if file.endswith(".pdf"):
#             files_list.append(os.path.join(root, file))
            
# for file in files_list:
#     print(file)


# In[4]:


# file_content = []
# for file in files_list:
#     content_data= ""
#     PDF_fileObj2 = open(file, 'rb')
#     pdfReader = PyPDF2.PdfFileReader(PDF_fileObj2)
#     for i in range(0 , pdfReader.numPages):
#         pageObj = pdfReader.getPage(i)
#         if i <=3:   #Extracting first 3 pages from PDF
#             content_text = pageObj.extractText()
#             content_data += content_text
#     file_content.append(content_data)


# In[4]:


# file_content


# In[5]:


#Exporting to excel
# pd.DataFrame(file_content).to_excel("")  #Export to excel file (specify path) to create training data


# In[56]:
#files_list=[]
for root, dirs, files in os.walk(""):   #filepath for retrained data (training data + feedback data).
    for file in files:
         if file.endswith(".xlsx"):
                print(file)
#               files_list.append(file)
                
text_data_df = pd.read_excel(os.path.join(root, file), index_col=[0])

# In[57]:


text_data_df.info()
text_data_df.head()


# In[58]:


#Stopwords
from nltk.corpus import stopwords
stop_words_full = pd.read_excel("") #Stop words file path (extracted from web)
stop_words_full_list = [i for i in stop_words_full['stop_words']]

stop_words = set(stop_words_full_list + stopwords.words('english'))

#Geo-words
geo_words = pd.read_excel("")  #Geo specific words filepath
geo_words = [i for i in geo_words['geo_words']]


# ### No of documents for each section:

# In[59]:


text_data_df['LABEL'].str.strip().value_counts()


# ### Removing punctuations and cleaning

# In[60]:


text_data_df['LABEL'] = text_data_df['LABEL'].str.strip()
punct = [p for p in set(string.punctuation) if p not in (".")]

for i in range(0,len(text_data_df['TEXT'])):
    if type(text_data_df.iloc[i]['TEXT']) != float:
        text_data_df.iloc[i]['TEXT'] = text_data_df.iloc[i]['TEXT'].lower().replace("\n"," ").replace("\t"," ").strip(" ")
        text_data_df.iloc[i]['TEXT'] = "".join(c for c in text_data_df.iloc[i]['TEXT'] if c not in punct)
        text_data_df.iloc[i]['TEXT'] = " ".join([c for c in text_data_df['TEXT'].iloc[i].split(" ") if not(c[:1].isdigit() and c[1:2] in (p for p in punct))])
        text_data_df.iloc[i]['TEXT'] = " ".join([w for w in text_data_df['TEXT'].iloc[i].split() if w not in stop_words])
        text_data_df.iloc[i]['TEXT'] = " ".join([w for w in text_data_df['TEXT'].iloc[i].split() if w[:-1] not in stop_words])
        text_data_df.iloc[i]['TEXT'] = " ".join([w for w in text_data_df['TEXT'].iloc[i].split() if w not in geo_words])  
        text_data_df.iloc[i]['TEXT'] = " ".join([w for w in text_data_df['TEXT'].iloc[i].split() if w[:1] not in list(map(lambda x: str(x),range(3))) and w[:1] not in list(map(lambda x: str(x),range(4,10)))])  
        text_data_df.iloc[i]['TEXT'] = " ".join([w for w in text_data_df['TEXT'].iloc[i].split(" ") if not(w[:1].isdigit() and w[1:].isalpha())])
        text_data_df.iloc[i]['TEXT'] = " ".join([w for w in text_data_df['TEXT'].iloc[i].split(" ") if not(w[:3].isdigit() and w[3:].isalpha())])
        text_data_df.iloc[i]['TEXT'] = " ".join([w[:-1] if not(w[:1].isdigit()) and w.endswith(".") else w for w in text_data_df.iloc[i]['TEXT'].split(" ")])
        text_data_df.iloc[i]['TEXT'] = " ".join([w for w in text_data_df['TEXT'].iloc[i].split(" ") if len(w) > 2 and len(w) < 15])
    
text_data_df.head()


# ### Cleaning II:

# In[61]:


#Removing dot(.) from text
for i in range(0,len(text_data_df['TEXT'])):
    if type(text_data_df.iloc[i]['TEXT']) != float:
        text_data_df.iloc[i]['TEXT'] = " ".join([w.replace("."," ") if len(w) > 9 or len(w) < 7 else w for w in text_data_df['TEXT'].iloc[i].split(" ") ])  


# ### Lemmatization

# In[62]:


tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
for i in range(len(text_data_df['TEXT'])):
    lemma = []
#     text_lemma = ""
    text_tokens = word_tokenize(text_data_df.iloc[i]['TEXT'])
    lemma_function = WordNetLemmatizer()
    for token , tag in pos_tag(text_tokens):
        lemma.append(lemma_function.lemmatize(token, tag_map[tag[0]]))
    text_data_df.iloc[i]['TEXT'] = " ".join(l for l in lemma )


# In[63]:


text_data_df.head()


# ### Extracting identified keywords from text

# In[64]:


Section_keywords = pd.read_excel("D:\\Nishant\\ML_Project\\m3\\ML Resources\\M3_Keywords_32S_individual_words_32S7.xlsx")
Keywords_df = Section_keywords.copy().fillna(0)
Keywords_df = Keywords_df.drop(Keywords_df.columns[:3],axis=1)
Keywords_df.head(20)


# In[65]:


keyword_list = []
for col in Keywords_df.columns:
    for val in Keywords_df[col]:
        if val != 0:
            keyword_list.append(val.lower())
            
#Keywords lemmatization            
keyword_lemma=[]
for token , tag in pos_tag(keyword_list):
        keyword_lemma.append(lemma_function.lemmatize(token, tag_map[tag[0]]))

print(keyword_list[:5])
print(keyword_lemma[:5])


# In[66]:


keyword_list_unique = set(keyword_lemma)
len(keyword_list_unique)


# In[14]:


# pd.DataFrame(keyword_list_unique).to_excel("D:\\Nishant\\ML_Project\\m3\\Training_Data\\keywords_list.xlsx")


# In[67]:


#Matching & Filtering data with keywords:
for i in range(len(text_data_df['TEXT'])):
    if type(text_data_df.iloc[i]['TEXT']) != float:
        text_data_df.iloc[i]['TEXT'] = " ".join([w for w in text_data_df.iloc[i]['TEXT'].split(" ") if w in keyword_list_unique])
    elif type(text_data_df.iloc[i]['TEXT']) == float:
        text_data_df.iloc[i]['TEXT'] = " "
        
text_data_df.head()


# ### Randomise data before train test split

# In[68]:


text_data_randomise = text_data_df.sample(frac=1).reset_index(drop=True)
text_data_randomise.head(10)


# In[69]:


np.random.seed(442)
X_train, X_test, y_train, y_test = train_test_split(text_data_randomise['TEXT'], text_data_randomise['LABEL'], 
                                                    test_size=0.2, random_state=1)
print(len(X_train),len(y_train),len(X_test),len(y_test))


# In[70]:


X_train.head()


# In[71]:


y_train[:5]


# In[72]:


y_test[:5]


# ### Feature engineering

# In[101]:


# Fit and tranform X_train
count_vectorizer = CountVectorizer(strip_accents='ascii',lowercase=True, analyzer='word', 
                                   max_df=0.25, min_df=0.05, ngram_range=(1, 2),
                                token_pattern= u'(?ui)\\b(?:3\.\w+)+(?:\.\w+)+\\b|\\b\\w*[a-zA-Z]+\\w*\\b')
                                   
X_train_cv = count_vectorizer.fit_transform(X_train)
# # Save vectorizer.vocabulary_
# pickle.dump(count_vectorizer.vocabulary_,open("D://Nishant//ML_Project//m3//Trained_Models/vocabulary_32S6_5SEP_FINAL.pkl","wb"))
import pickle
rand_num = np.random.randint(212)
root_path = "D:/Nishant/ML_Project/m3/Trained_Models/" 
pickle.dump(count_vectorizer.vocabulary_,open(root_path + "vocab_" + str(rand_num) + ".pkl","wb"))

print ('Shape of Sparse Matrix: ', X_train_cv.shape)
print ('Amount of Non-Zero occurences: ', X_train_cv.nnz)
print ('sparsity: %.2f%%' % (100.0*X_train_cv.nnz/ (X_train_cv.shape[0] * X_train_cv.shape[1])))

# Transform X_test
X_test_cv = count_vectorizer.transform(X_test)


# In[102]:


Features = pd.DataFrame(count_vectorizer.get_feature_names())
Features.head(10)


# ### Model Building

# In[103]:


NB_Model = MultinomialNB(alpha=0.01)
NB_Model.fit(X_train_cv.toarray(), np.array(y_train))


# In[104]:


print(y_test)


# In[105]:


y_pred = NB_Model.predict(X_test_cv.toarray())
y_pred


# In[106]:


ticks = ['3.2.S.1.1','3.2.S.1.2','3.2.S.1.3','3.2.S.2.1','3.2.S.2.2','3.2.S.2.3','3.2.S.2.4','3.2.S.2.5','3.2.S.2.6',
         '3.2.S.3.1', '3.2.S.3.2', '3.2.S.5', '3.2.S.6']


# In[107]:


acc_score = accuracy_score(y_test,y_pred)
acc_score


# ### Confusion Matrix:

# In[108]:


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, square=False, annot=True, annot_kws={"size": 14}, xticklabels=ticks, yticklabels=ticks, cmap='coolwarm', cbar=False)
sns.set(font_scale=1)
plt.xlabel('PREDICTED CLASS')
plt.ylabel('ACTUAL CLASS')
#plt.show()


# ### Classification report:

# In[109]:


print('\nClasification report:\n', classification_report(y_test, y_pred))


# ### Class Probabilities:

# In[110]:


probabilites = NB_Model.predict_proba(X_test_cv.toarray())
for j in range(len(probabilites)):
        print([round(i*100,2) for i in probabilites[j]])


# ### save the model to disk

# In[111]:

#filename = "D:\\Nishant\\ML_Project\\m3\\Trained_Models\\NB_model_9SEP_32S7.mdl"
rand_num_model = np.random.randint(22)
filename = "D:\\Nishant\\ML_Project\\m3\\Trained_Models\\" + "NB_Model_" + str(rand_num_model) + ".mdl"
#print(filename)
pickle.dump(NB_Model, open(filename, 'wb'))

# ### load the model from disk

# In[32]:


model_loaded = pickle.load(open(filename, 'rb'))
model_loaded


# ### loading vocab

# In[65]:


#vocab = "D://Nishant//ML_Project//m3//Trained_Models/vocabulary_9SEP_32S7.pkl"
loaded_vectorizer = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open(root_path + "vocab_" + str(rand_num) + ".pkl","rb")))


# ### Test document (test case)

# In[87]:


testfiles=[]
for root, dirs, files in os.walk("D://Nishant//ML_Project//m3//Training_data_Rathnadeep/Set-2/"):
    for file in files:
        if file.endswith(".pdf"):
            testfiles.append(os.path.join(root, file))
            
testfile_content = []
for file in testfiles[:5]:
    print(file)
    content_data= ""
    PDF_fileObj2 = open(file, 'rb')
    pdfReader = PyPDF2.PdfFileReader(PDF_fileObj2)
    for i in range(0 , pdfReader.numPages):
        pageObj = pdfReader.getPage(i)
        if i <=5:
            content_text = pageObj.extractText()
            content_data += content_text
    testfile_content.append(content_data.replace("\n"," "))
    
test_data = pd.DataFrame(testfile_content)
test_data.columns = [['TEXT']]
#test_data.info()


# In[55]:

for i in range(0,len(test_data['TEXT'])):
    test_data.iloc[i]['TEXT'] = test_data.iloc[i]['TEXT'].str.lower().replace("\n"," ").replace("\t"," ").str.strip(" ")
    test_data.iloc[i]['TEXT'] = "".join(c for c in test_data.iloc[i]['TEXT'] if c not in punct)

test_data.head(2)


# ### Prediction

# In[132]:


for i in range(len(test_data)):
    test_cv = loaded_vectorizer.transform(test_data.iloc[i])
    print("Predicted Label:" + "\tDoc " + str(i) + "\t" + str(model_loaded.predict(test_cv.toarray())))
    print("Predicted Probability:\t\t" + str(np.max(model_loaded.predict_proba(test_cv.toarray()))*100))

