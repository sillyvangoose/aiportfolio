#!/usr/bin/env python
# coding: utf-8

# # Critically Engaging with AI Ethics
# 
# In this lab we will be critically engaging with existing datasets that have been used to address ethics in AI. In particular, we will explore the [**Jigsaw Toxic Comment Classification Challenge**](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge). This challenge brought to light bias in the data that sparked the [Jigsaw Unintended Bias in Toxicity Classification Challenge](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification). 
# 
# In this lab, we will dig into the dataset ourselves to explore the biases. We will further explore other datasets to expand our thinking about bias and fairness in AI in relation to aspects such as demography and equal opportunity as well as performance and group unawareness of the model. We will learn more about that in the tutorial below.
# 
# # Task 1: README!
# 
# This week, coding activity will be minimal, if any. However, as always, you will be expected to incorporate your analysis, thoughts and discussions into your notebooks as markdown cells, so I recommend you start up your Jupyter notebook in advance. As always, **remember**:
# 
# - To ensure you have all the necessary Python libraries/packages for running code you are recommended to use your environment set up on the **Glasgow Anywhere Student Desktop**.
# - Start anaconda, and launch Jupyter Notebook from within Anaconda**. If you run Jupyter Notebook without going through Anaconda, you might not have access to the packages installed on Anaconda.
# - If you run Anaconda or Jupyter Notebook on a local lab computer, there is no guarantee that these will work properly, that the packages will be available, or that you will have permission to install the extra packages yourself.
# - You can set up Anaconda on your own computer with the necessary libraries/packages. Please check how to set up a new environement in Anaconda and review the minimum list of Python libraries/packages, all discussed in Week 4 lab.
# - We strongly recommend that you save your notebooks in the folder you made in Week 1 exercise, which should have been created in the University of Glasgow One Drive - **do not confuse this with personal and other organisational One Drives**. Saving a copy of your notebooks on the University One Drive ensures that it is backed up (the first principles of digital preservation and information mnagement).
# - When you are on the Remote desktop, the `University of Glasgow One Drive` should be visible in the home directory of the Jupyter Notebook. Other machines may require additional set up and/or navigation for One Drive to be directly accessible from Jupyter Notebook.
# 

# # Task 2: Identifying Bias
# 
# This week we will make use of one of the [Kaggle](https://www.kaggle.com) tutorials and their associated notebooks to learn how to identify different types of bias. Biases can creep in at any stage of the AI task, from data collection methods, how we split/organise the test set, different algorithms, how the results are interpreted and deployed. Some of these topics have been extensively discussed and as a response, Kaggle has developed a course on AI ethics:
# 
# - Navigate to the [Kaggle tutorial on Identifying Bias in AI](https://www.kaggle.com/code/alexisbcook/identifying-bias-in-ai/tutorial). 
# - In this section we will explore the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge) to discover different types of biases that might emerge in the dataset. 
# 
# #### Task 2-a: Understanding the Scope of Bias
# 
# Read through the first page of the [Kaggle tutorial on Identifying Bias in AI] to understand the scope of biases discussed at Kaggle.
# - How many types of biases are described on the page? 
# - Which type of bias did you know about already before this course and which type was new to you? 
# - Can you think of any others? Create a markdown cell below to discuss your thoughts on these questions.
# 
# Note that the biases discussed in the tutorial are not an exhaustive list. Recall that biases can exist across the entire machine learning pipeline. 
# 
# - Scroll down to the end of the Kaggle tutorial page and click on the link to the exercise to work directly with a model and explore the data.** 
# 
# #### Task 2-b: Run through the tutorial. Take selected screenshorts of your activity while doing the tutorial.
# 
# - Discuss with your peer group, your findings about the biases in the data, including types of biases. 
# - Demonstrate your discussion with examples and screenshots of your activity on the tutorial. Present these in your own notebook.
# 
# Modify the markdown cell below to address the Tasks 2-a and 2-b.

# **Markdown for discussing bias**
# 
# 2.a: there are six kinds of bias. I was familiar with representation bias from statistics in high school, and historical bias is new to me, but makes loss of sense. I know that there are more, for example comfirmation bias, which is the creator's bias creeping into data.
# 
# 2.b: the notebook showed us all six kinds of bias, but I personally found the historical bias the most interesting. I also found that while this may not be the goal of the ai, it was completely unable to pick up on nuances of toxic comments, such as sarcasm. 
# 

# In[1]:


from IPython.display import Image


# In[2]:


Image ("sarcasm.png")


# In[3]:


Image ("toxic.png")


# # Task 3: Large Language Models and Bias: Word Embedding Demo
# 
# Go to the [embedding projector at tensorflow.org](http://projector.tensorflow.org/). This may take some time to load so be patient! There is a lot of information being visualised. This will take especially long if you select "Word2Vec All" as your dataset. The projector provides a visualisation of the langauge language model called **Word2Vec**.
# 
# This tool also provides the option of visualising the organisation of hand written digits from the MNIST dataset to see how data representations of the digits are clustered together or not. There is also the option of visualising the `iris` dataset from `scikit-learn` with respect to their categories. Feel free to explore these as well if you like.
# 
# For the current exercise, we will concentrate on exploring the relationships between the words in the **Word2Vec** model. First, select **Word2Vec 10K** from the drop down menu (top lefthand side). This is a reduced version of **Word2Vec All**. You can search for words by submitting them in the search box on the right hand side. 
# 
# #### Task 3.1: Initial exploration of words and relationships
# 
# - Type `apple` and click on `Isolate 101 ppints`. This reduces the noise. Note how juice, fruit, wine are closer together than macintosh, computers and atari. 
# - Try also words like `silver` and `sound`. What are your observations. Does it seem like words related to each other are sitting closer to each other?
# 
# #### Task 3.2: Exploring "Word2Vec All" for patterns
# 
# - Try to load "Word2Vec All" dataset if you can (this may take a while so be patient!) and explore the word `engineer`, `drummer`or any other occupation - what do you find? 
# - Do you think perhaps there are concerns of gender bias? If so, how? If not, why not? Discuss it with our peer group and present the results in a your notebook.
# - Why not make some screenshots to embed into your notebook along with your comment? This could make it more understandable to a broader audience. 
# - Do not forget to include attribution to the authors of the Projector demo.
# 
# Modify the markdown cell below to present your thoughts.

# **Markdown cell for discussing large language models**
# 
# 3.1 I would think that the closer the word would be the more related it would be but metal is very far away from silver, amking me think that there is another variable in the mix.
# 
# 3.2 I looked up politician, I found a list of other careers, which I've found to be relatively unrelated to politician.
# I've not really found gender bias: the closest careers were statesman and footballer which are generally associated with men, but the third one down is actress which is associated with women, so I think there is a relative even spread, but there could be an argument made for men being more closely associated with politicians than women.
# 
# I could not find the names of the people who created the demo but the open documentation link on the demo has an article and the page is licensed under the Creative Commons Attribution 4.0 License, so I guess that is all I can attribute to.
# 

# In[5]:


Image ("politician.png")


# In[6]:


Image ("careers.png")


# # Task 4: Thinking about AI Fairness 
# 
# So we now know that AI models (e.g. large language models) can be biased. We saw that with the embedding projector already. We discussed in the previous exercise about the machine learning pipeline, how the assessment of datasets can be crucicial to deciding the suitability of deploying AI in the real world. This is where data connects to questions of fairness.
# 
# - Navigate to the [Kaggle Tutorial on AI Fairness](https://www.kaggle.com/code/alexisbcook/ai-fairness). 
# 
# #### Task 4-a: Topics in AI Fairness
# Read through the page to understand the scope of the fairness criteria discussed at Kaggle. Just as we dicussed with bias, the fairness criteria discussed at Kaggle is not exhaustive. 
# - How many criteria are described on the page? 
# - Which criteria did you know about already before this course and which, if any, was new to you? 
# - Can you think of any other criteria? Create a markdown cell and note down your discussion with your peer group on these questions.
# 
# #### Task 4-b: AI fairness in the context of the credit card dataset. 
# Scroll down to the end of [the page on AI fairness](https://www.kaggle.com/code/alexisbcook/ai-fairness) to find a link to another interactive exercise to run code in a notebook using credit card application data.
# - Run the tutorial, while taking selected screenshots.
# - Discuss your findings with your peer group.
# - Note down the key points of your activity and discussion in your notebook using the example and screenshots of your activity on the tutorial.
# 
# 
# Report the results of the activity and discussion by modifying the markdown cell below.

# **Markdown cell for discussing fairness**
# 4.a: There are four criteria, and I knew about Group Unawareness. It reminds me a lot of blind and double blind testing in experiements. I did not know about demographic parity.
# 
# 4.b: I found the demographic unfairness to be particularly interesting. On paper I suppose it would be easy to have a 50/50 split from Groups A and B, but each sample from each group aren't exactly the same so you can't have that even spread every time you run a different test.
# 

# In[7]:


Image ("check.png")


# In[8]:


Image ("flowchart.png")


# # Task 5: AI and Explainability
# 
# In this section we will explore the reasons behind decisions that AI makes. While this is really hard to know, there are some approaches developed to know which features in your data (e.g. median_income in the housing dataset we used before) played a more important role than others in determining how your machine learning model performs. One of the many approaches for assessing feature importance is **permutation importance**.
# 
# The idea behind permutation importance is simple. Features are what you might consider the columns in a tabulated dataset, such as that might be found in a spreadsheet. 
# - The idea of permutation importance is that a feature is important if the performance of your AI program gets messed up by **shuffling** or **permuting** the order of values in that feature column for the entries in your test data. 
# - The more your AI performance gets messed up in response to the shuffling, the more likely the feature was important for the AI model.
#  
# To make this idea more concrete, read through the page at the [Tutorial on Permutation Importance](https://www.kaggle.com/code/dansbecker/permutation-importance) at Kaggle. The page describes an example to "predict a person's height when they become 20 years old, using data that is available at age 10". 
# 
# The page invites you to work with code to calculate the permutation importance of features for an example in football to predict "whether a soccer/football team will have the "Man of the Game" winner based on the team's statistics". Scroll down to the end of the page to the section "Your Turn" where you will find a link to an exercise to try it yourself to calculate the importance of features in a Taxi Fare Prediction dataset.
# 
# #### Task 1-a: Carry out the exercise, taking screenshots of the exercise as you make progress. Using screen shots and text in your notebook, answer the following question: 
# 1. How many features are in this dataset? 
# 2. Were the results of doing the exercise contrary to intuition? If yes, why? If no, why not? 
# 3. Discuss your results with your peer group.
# 4. Include your screenshots, text, and discyssions in a markdown cell.
# 
# #### Task 1-b: Reflecting on Permutation Importance.
# 
# - Do you think the permutation importance is a reasonable measure of feature importance? 
# - Can you think of any examples where this would have issues? 
# - Discuss these questions in your notebook - describe your example, if you have any, and discuss the issues. 

# 1.a There are five features, and it was contrary to what I thought it was going to be. I did agree with the notebook in thinking that each feature would be equally important, and I was surprised that latitude was more imprtant than longitude.
# 1.b I don't think permutation importance is a reasonable measure of feature importance, as in Question 6, the permutation importance couldn't imply anything about the expensiveness of the features, which could cause issues by making it harder to check feature importance.

# In[9]:


Image ("table.png")


# # Task 6: Further Activities for Broader Discussion
# 
# Apart from the [**Jigsaw Toxic Comment Classification Challenge**](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge) another challenge you might explore is the [**Inclusive Images Challenge**](https://www.kaggle.com/c/inclusive-images-challenge). Read at least one of the following.
# 
# - The [announcement of the Inclusive Images Challenge made by Google AI](https://ai.googleblog.com/2018/09/introducing-inclusive-images-competition.html). Explore the [Open Images Dataset V7](https://storage.googleapis.com/openimages/web/index.html) - this is where the Inclusive Images Challenge dataset comes from.
# - Article summarising [the Inclusive Image Challenge at NeurIPS 2018 conference](https://link.springer.com/chapter/10.1007/978-3-030-29135-8_6)
# - Explore the [recent controversy](https://www.theverge.com/21298762/face-depixelizer-ai-machine-learning-tool-pulse-stylegan-obama-bias) about bias in relation to [PULSE](https://paperswithcode.com/method/pulse) which, among other things, sharpens blurry images.
# - Given your exploration in the sections above, what problems might you foresee with [these tasks attempted with the Jigsaw dataset on toxicity](https://link.springer.com/chapter/10.1007/978-981-33-4367-2_81)?
# 
# There are many concepts (e.g. model cards and datasheets) omitted in discussion above about AI and Ethics. To acquire a foundational knowledge of transparency, accessibility and fairness:
# 
# - You are welcome to carry out the rest of the [Kaggle course on Intro to AI Ethics](https://www.kaggle.com/learn/intro-to-ai-ethics) to see some ideas from the Kaggle community. 
# - You are welcome to carry out the rest of the [Kaggle tutorial on explainability]( https://www.kaggle.com/learn/machine-learning-explainability) but these are a bit more technical in nature.
# - 

# # Summary
# 
# In this lab, you explored a number of areas that pose challenges with regard to AI and ethics: bias, fairness and explainability. This, and other topics in reposible AI development, is currently at the forefront of the AI landscape. 
# 
# The discussions coming up in the lectures on applications of AI (to be presented by guest lecturers in the weeks to come) will undoubtedly intersect with these concerns. In preparation, you might think, in advance, about **what distinctive questions about ethics might arise in AI applications in law, language, finance, archives, generative AI and beyond**.   
