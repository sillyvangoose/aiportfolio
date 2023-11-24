#!/usr/bin/env python
# coding: utf-8

# # Week 1: Getting Started with Anaconda, Jupyter Notebook and Python
# A: Why I joined the course, for motivation, vision, aspiration? Aspiration!
# B: No real prior experience with AI beyond encountering it in every day life.
# C: Just in general more about AI, I want to immerse myself in it and branch out my knowledge!
# message + message prints it twice with no space, message*3 does it three times with no spaces in between, message in brackets 0 prints only the first letter, i think bc each number is associated with the order of the letter (0 is first letter 1 is second letter)
# I think message is a good variable name for the time being, but it should definitely change to fit the situation as time goes on

# In[1]:


variable = "buying my wii was a mistake"
print (variable)


# In[2]:


from IPython.display import *


# In[3]:


YouTubeVideo("bsyY9m7Q2KI?si=dSl1FCb21rM7DkUM")


# # Week 2: Exploring Data 

# In[2]:


from IPython.display import Image


# In[4]:


Image ("picture1.jpg")


# In[6]:


from IPython.display import Audio


# In[ ]:


Audio ("audio1.mid")


# In[ ]:


Audio ("audio2.ogg")
#This file is licensed under the Creative Commons Attribution-Share Alike 3.0 Unported license. You are free: to share – to copy, distribute and transmit the work, to remix – to adapt the work, Under the following conditions: attribution – You must give appropriate credit, provide a link to the, license, and indicate if changes were made. You may do so in any, reasonable manner, but not in any way that suggests the licensor endorses you or your use. share alike – If you remix, transform, or build upon the material, you must distribute your contributions under the same or compatible license as the original. The original ogg file was found at the url: https://en.wikipedia.org/wiki/File:GoldbergVariations_MehmetOkonsar1of3_Var1to10.ogg


# Only audio 2 played, I think this might be because Jupyter doesn't support .mid files? Also I'm not sure where to put the attribution so I put it above in the code.

# # Week 2 Part 2: madplotlib

# In[5]:


from matplotlib import pyplot
test_picture = pyplot.imread("picture1.jpg")
print("Numpy array of the image: ", test_picture)
pyplot.imshow(test_picture)


# I'm not exactly sure what's going on here, but maybe the series of numbers are R, B, G arrays for the colors of the pixels in the photo?

# # Week 2 Part 3: scikit-learn

# In[ ]:


from sklearn import datasets
dir (datasets)


# I'm picking load_wine and load_iris because I am curious as to what data is associated with wine and iris, what exactly is it measuring?

# In[ ]:


wine_data = datasets.load_wine()


# In[ ]:


wine_data.DESCR


# In[ ]:


print(wine_data.DESCR)


# In[ ]:


iris_data = datasets.load_iris()


# In[ ]:


iris_data.DESCR


# In[ ]:


print(iris_data.DESCR)


# In[ ]:


iris_data.feature_names


# In[ ]:


iris_data.target_names


# In[ ]:


wine_data.feature_names


# In[ ]:


wine_data.target_names


# # Week 2: Pandas!

# In[ ]:


from sklearn import datasets
import pandas
wine_data = datasets.load_wine()
wine_dataframe = pandas.DataFrame(data=wine_data['data'], columns = wine_data['feature_names'])


# In[ ]:


wine_dataframe.head()
wine_dataframe.describe()


# I think these commands assign labels/descriptions of the data, on the top and side row, assigning meaning to the information in the table

# # Week 3: Thinking About Data Bias

# Article by Prabhakar Krishnamurthy thoughts below:
# I think it is very generous to say that most ML modelers don't realise the bias is happening or they do and don't know what to do about it. That is not backed up with any evidence, and no sources are cited. Additionally, ML modelers not noticing that there is a bias is a bias in itself, they are priveledged enough to not think about racism or sexism, as the majority of people involved in computer science are white and male, so they do not think about bias being there. Additionally, if most ML modelers don't know it's there, then why is the first part of the article a list of instances of bias that are backed up by evidence? I'm more inclined to agree with the instances backed up by evidence that there has been bias rather than speculation/opinions that bias does not happen.

# In[ ]:


get_ipython().system('pipreqs')

