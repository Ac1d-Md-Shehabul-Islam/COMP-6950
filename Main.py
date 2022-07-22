#!/usr/bin/env python
# coding: utf-8

# ## Review Analysis of Amazon using Text Classification ##

# ### Mohammad Shehabul Islam
# ### Student ID - 202196528

# In[1]:


get_ipython().run_line_magic('run', 'Class.ipynb')


# In[2]:


d = readdata('/Users/Siam Islam/Desktop/mohammadsi_6950_project/dataset/csv/book.csv')
df = d.getDataFrame()
df.head(3)


# In[3]:


df = d.dropColumn(df)
df


# In[4]:


p = preprocess(df)
ren = p.rename()
ren


# In[5]:


df = p.addLabel()
df


# In[6]:


v = view(df)
v.plot_rating()


# In[7]:


v.plot_label()


# In[8]:


p.getTextLen()


# In[9]:


p.getPuncPerc()


# In[10]:


p.getCleanText()


# In[11]:


p.tokenizedCleanText()


# In[12]:


p.textNoStopWord()


# In[13]:


p.ps_stem()


# In[14]:


p.wn_lemm()


# In[15]:


p.processText()


# In[16]:


v.wordCompare()


# In[17]:


v.puncCompare()


# In[18]:


trte = train_test(df)


# In[19]:


df


# In[20]:


c,d = trte.split()


# In[21]:


a,b = trte.tfidf_vec()


# In[22]:


a


# In[23]:


b


# In[24]:


m = model(df)


# In[25]:


m.naive_bayes(a, b, c, d)


# In[26]:


m.random_forest(a, b, c, d)

