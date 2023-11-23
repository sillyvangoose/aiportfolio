#!/usr/bin/env python
# coding: utf-8

# # Generating Text with Neural Networks
# 

# ## Description
# This neural network, specifically a Recurrent Neural Network, was created by Andrej Karpathy. The data is a combination of excerpts from several Shakespeare plays, the example given being from Shakespeare's last political play, Coriolanus. This neural network uses this large supply of Shakespeare to generate text that imitates Shakespeare's works.

# ## Predicted Purpose
# I can find many uses for an AI such as this. Creating Shakespeare adjacent plays could be helpful for a theatre company, looking to put on a parody of a Shakespeare show. It could also be used in theatre education as an introduction to Shakespeare. Some find Shakespeare very hard to deconstruct, so having text in the style of Shakespeare to deconstruct first might be helpful to some student instead of looking at actual Shakespeare.
# It could also be used as introduction to AI for theatre students who know less about the field. Among the theatre community and other people involved in fine arts, AI is thought to be only used for STEM subjects, more specifically with computer science. It is something that is not accessible or for them, and an AI model like this could show them that AI is for everyone.

# # Getting the Data

# In[1]:


import tensorflow as tf

shakespeare_url = "https://homl.info/shakespeare"  # shortcut URL
filepath = tf.keras.utils.get_file("shakespeare.txt", shakespeare_url)
with open(filepath) as f:
    shakespeare_text = f.read()


# This cell above is importing the data from the page homl.info/shakespeare. If you type this link into your browser to look at what it is, it is a VERY long page filled with text from Shakespeare plays. This is the data our neural network will use to train to later produce text. It can be thought of as a guide for the neural network for its output.

# In[2]:


print(shakespeare_text[:80]) # not relevant to machine learning but relevant to exploring the data


# The cell above is printing part of that data. It has printed the first part of the data, and I think this can be useful to check that the data has been imported correctly, but also to get a sample of your data.

# # Preparing the Data

# In[3]:


text_vec_layer = tf.keras.layers.TextVectorization(split="character",
                                                   standardize="lower")
text_vec_layer.adapt([shakespeare_text])
encoded = text_vec_layer([shakespeare_text])[0]


# In[4]:


print(text_vec_layer([shakespeare_text]))


# In[5]:


encoded -= 2  # drop tokens 0 (pad) and 1 (unknown), which we will not use
n_tokens = text_vec_layer.vocabulary_size() - 2  # number of distinct chars = 39
dataset_size = len(encoded)  # total number of chars = 1,115,394


# In[6]:


print(n_tokens, dataset_size)


# In[7]:


def to_dataset(sequence, length, shuffle=False, seed=None, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window_ds: window_ds.batch(length + 1))
    if shuffle:
        ds = ds.shuffle(100_000, seed=seed)
    ds = ds.batch(batch_size)
    return ds.map(lambda window: (window[:, :-1], window[:, 1:])).prefetch(1)


# In[8]:


length = 100
tf.random.set_seed(42)
train_set = to_dataset(encoded[:1_000_000], length=length, shuffle=True,
                       seed=42)
valid_set = to_dataset(encoded[1_000_000:1_060_000], length=length)
test_set = to_dataset(encoded[1_060_000:], length=length)


# The cells in this section are prepping the data to be used. It seperates the data into appropriate collections which are identified as chars (character data). After that seperation we are told the size of the data in that separation. There are 39 distinct characters and 1,115,394 total characters.

# # Building and Training the Model

# The cell below is used to train the text generation model. Tensorflow Keras Sequential specifies layers in the neural network, kind of laying out a map for steps in between input and output. The layers (or steps) are Embedding, GRU, and Dense. GRU is a gated recurrent unit. I'm not sure why, but with each epoch, there is a warning about certain gru functions. I'm not sure how to fix this, and I can't tell that this is effecting text generation. 
# Each epoch is all of the training data going through the algorithm at once. There are 10 iderations of this, as defined in the cell below. The result of each epoch will have speed per step, loss and accuracy. Loss determines the inaccuracy of results during a test, and the accuracy is the opposite.

# In[9]:


tf.random.set_seed(42)  # extra code – ensures reproducibility on CPU
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=n_tokens, output_dim=16),
    tf.keras.layers.GRU(128, return_sequences=True),
    tf.keras.layers.Dense(n_tokens, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
model_ckpt = tf.keras.callbacks.ModelCheckpoint(
    "my_shakespeare_model", monitor="val_accuracy", save_best_only=True)
history = model.fit(train_set, validation_data=valid_set, epochs=10,
                    callbacks=[model_ckpt])


# In[10]:


shakespeare_model = tf.keras.Sequential([
    text_vec_layer,
    tf.keras.layers.Lambda(lambda X: X - 2),  # no <PAD> or <UNK> tokens
    model
])


# # Generating Text

# The cells below are used to generate text in the style of Shakespeare. I'm interpreting the cell directly below as a control variable to make sure the code is working, by having the text being generated be one letter, which, if the code is working correctly, will be e. The cell below that one looks at shape and dimension size.
# The last three cells are actually generating the text by using the print function. The command it uses next to print is extend, so the AI is finishing the phrase 'To be or not to be' in the style of Shakespeare. The temperature is changed in each line, resulting in a different and progressively less accurate, with the last not even being a sentence. Temperature does exactly this, it determines the output's accuracy during text generation. The higher it is, the worse the output is. I'm not sure what the purpose of raising the temperature would be beyond experimenting for fun. Maybe one would use a slightly higher temperature to parody and make fun of Shakespeare? 

# In[11]:


y_proba = shakespeare_model.predict(["To be or not to b"])[0, -1]
y_pred = tf.argmax(y_proba)  # choose the most probable character ID
text_vec_layer.get_vocabulary()[y_pred + 2]


# In[12]:


log_probas = tf.math.log([[0.5, 0.4, 0.1]])  # probas = 50%, 40%, and 10%
tf.random.set_seed(42)
tf.random.categorical(log_probas, num_samples=8)  # draw 8 samples


# In[13]:


def next_char(text, temperature=1):
    y_proba = shakespeare_model.predict([text])[0, -1:]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1)[0, 0]
    return text_vec_layer.get_vocabulary()[char_id + 2]


# In[14]:


def extend_text(text, n_chars=50, temperature=1):
    for _ in range(n_chars):
        text += next_char(text, temperature)
    return text


# In[15]:


tf.random.set_seed(42)  # extra code – ensures reproducibility on CPU


# In[16]:


print(extend_text("To be or not to be", temperature=0.01))


# In[17]:


print(extend_text("To be or not to be", temperature=1))


# In[19]:


print(extend_text("To be or not to be", temperature=100))


# # More Interpretation
# What Hugging Chat produced:
# To be or not to be, that is the query, 
# Whether tis better to exist, or not to be
# Hugging Chat is probably a more advanced AI and has a larger repository of data relating to Shakespeare. Our model only gives the model bits and pieces of different plays, while Hugging Chat most likely has access to the entirety of all Shakespeare plays. This is probably why, when I asked Hugging Chat to finish the phrase 'To be or not to be' in the style of Shakespeare (but not actually finishing the quote in actuality), it essentially reworded the 'To be or not to be' soliloqy. When I asked it to simply finish the phrase, without specifying that I didn't want it to be quoting Hamlet, it did actually finish the phrase from the script of Hamlet. This could be a wanted outcome for academic purposes. Say you're too lazy to look up the script of Hamlet, or you're worried you'll get too many different results and you don't have the time to scour the internet, you could turn to a text-generating AI to finish the monologue for you, like Hugging Chat succesfully did. Maybe this is what our AI aims to be with more training.
# As a side note, with our AI model, since the only text we're training it on is Shakespeare plays, we don't have to ask it to finish the phrase in the style of Shakespeare, and it finishes the phrase for more than three words. When I asked Hugging Chat to simply finish the phrase (without saying the style of Shakespeare), it returned "To be or not to be, that is the question." - Hamlet Act 3 Scene 1, which is true, but doesn't do exactly what I want it to do.

# In[ ]:




