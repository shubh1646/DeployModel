#!/usr/bin/env python
# coding: utf-8

# In[11]:


from interpret_community.widget import ExplanationDashboard
from sklearn.externals import joblib
from sklearn.model_selection import KFold, cross_val_score, train_test_split


# In[2]:


m = joblib.load("displaymodel.pkl")


# In[3]:


X_train=joblib.load("X_train")


# In[4]:


y_test = joblib.load("y_test")


# In[5]:


featu_na = ['Recency', 'Frequency', 'MonetaryValue', 'T-Age', 'RFM_Score',
       'Transaction', 'Revenue', 'avg_order_value', 'cust_lifetime_value',
       'R_4', 'R_3', 'R_2', 'R_1', 'F_1', 'F_2', 'F_3', 'F_4', 'M_1', 'M_2',
       'M_3', 'M_4']


# In[6]:


classes = ['Bronze','Silver','Gold']


# In[8]:


from sklearn.datasets import load_iris
from sklearn import svm

# Explainers:
# 1. SHAP Tabular Explainer
from interpret.ext.blackbox import TabularExplainer

# OR

# 2. Mimic Explainer
from interpret.ext.blackbox import MimicExplainer
# You can use one of the following four interpretable models as a global surrogate to the black box model
from interpret.ext.glassbox import LGBMExplainableModel
from interpret.ext.glassbox import LinearExplainableModel
from interpret.ext.glassbox import SGDExplainableModel
from interpret.ext.glassbox import DecisionTreeExplainableModel

# OR

# 3. PFI Explainer
from interpret.ext.blackbox import PFIExplainer


# In[9]:


explainer = TabularExplainer(m, 
                             X_train, 
                             features=featu_na,classes=classes)


# In[13]:


X_tests=joblib.load("X_tests")


# In[15]:


X_test=joblib.load("X_test")


# In[16]:


X_t, X_tests, y_train, y_tests = train_test_split(X_test, y_test, test_size=0.02, random_state=56)


# In[17]:


global_explanation = explainer.explain_global(X_tests)


# In[18]:


ExplanationDashboard(global_explanation, m, datasetX=X_tests)


# In[ ]:




