# Seq2Seq-chatbot
An implementation of a chatbot that uses a sequence to sequence (seq2seq) model which can be trained on any set of question-answer pairs. 

**Indented Usage**: With this script, one could try out different data sets and model parameters to see how the resulting models/chatbots differ.

**Script Summary**:
The goal of the script was to provide an easier way to vary various parameters regarding the seq2seq model (such as encoder & decoder recurrent steps, latent dimensions, training epochs, etc..). Furthermore, the script uses a more memory friendly way of training the seq2seq model, which allows the user to train the model with large data sets of question-answer pairs. Also, the script has some caching implemented since "preparing" the training data takes a long time for large datasets. Lastly, the script can save models once it's trained, as well as load trained models to chat with.

## Model Explanation
<p align="center">
  <img src="https://isaacchanghau.github.io/img/nlp/seq2seq-neuralconver/seq2seq.png" width="900">
  <br><i>Sequence to sequence model diagram.</i>
</p>

The underlying model of the ChatBot is a sequence to sequence model which is explained in detail in the following paper: [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf). But at a very high level, it encodes a question (a 'chat' input from the user) using an encoder LSTM to form a thought vector to 'describe' the question as a single vector. Then, the decoder (also a LSTM) takes said thought vector as the initial input and generates a response/answer. The picture above is a good visualization of the model.

**Implementation Details:**
The LSTMs required for the seq2seq model was implemented using [Keras](https://keras.io/) for simplicity. Furthermore, said model uses one-hot encodings of sentences its inputs and outputs. The one hot encodings use a vocab of the most frequent words/tokens of a document. Also, name entity recognition can be used (toggled) in this script and if it is enabled, all questions and answers will effectively have their entities subbed out for their 'entity tokens' before they are one hot encoded. For example, the sentence: "She was born on September 1st, 1996" would effectively be "She was born on <DATE> <DATE> <DATE>" before it is one hot encoded. The NER used in this script is from the [spaCy NLP library](https://spacy.io/). Note that NER considerably increases the time it takes to "prepare" the training data. 

## Features
**Veriable Model Parameter:**

**Training Memory Efficiency:**

**Data Caching:**

**Model Saving:**

## User Guide
**Dependencies:**

### Data & Vocab file details and spec

### Script options

## Cornell Movie Dialogs Results

## Credits
This script originally started off as a project assignment for [David Bamman's](http://people.ischool.berkeley.edu/~dbamman/) Natural Language Processing Course (Info 159/259) at UC Berkeley (with the goal of just creating a seq2seq chatbot). However the caching, model saving, data filtering, and parameter variations were all added after the assignment was submitted so that one could see how different model variations would perform.