
# coding: utf-8

# In[20]:


import pandas as pd
import joblib
import re


# In[104]:


file=open("input.txt","r",encoding='utf-8')
text=file.read()
file.close()
#text="@MoerasGrizzly the sourcelist is now dynamically generated. it's not that different from the github list though. :)"
print("\nInput:\n",text)


# In[105]:


df=pd.DataFrame({'text':[text]})
x=df['text']
x


# In[106]:


def preprocess_text(sen):
    sentence = remove_tags(sen)
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence


# In[107]:


TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_RE.sub('', text)


# In[108]:


X = []
sentences = list(df['text'])
for sen in sentences:
    X.append(preprocess_text(sen))


# In[109]:


X


# In[110]:


file=open("tokenizer.pickle","rb")
tokenizer=joblib.load(file)
file.close()


# In[111]:


file=open("labels.pkl","rb")
le=joblib.load(file)
file.close()


# In[112]:


file=open("model.pkl","rb")
classifier=joblib.load(file)
file.close()


# In[113]:


x_test=tokenizer.transform(X)


# In[114]:


y_pred= classifier.predict(x_test)


# In[115]:


y_pred


# In[116]:


out=le.inverse_transform(y_pred)
print(le.classes_)


# In[117]:


print("\nOutput:\n",out[0])


# In[118]:


file=open("output.txt","w")
file.write(out[0])
file.close()

