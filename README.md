# Seq2Seq-chatbot
An implementation of a chatbot that uses a sequence to sequence (seq2seq) model which can be trained on any set of question-answer pairs. 

[![dep0](https://img.shields.io/badge/Python-3.6%2B-brightgreen.svg)](https://www.python.org/downloads/) [![dep2](https://img.shields.io/badge/Keras-2.0+-blue.svg)](https://keras.io/) [![dep3](https://img.shields.io/badge/spaCy-2.0%2B-blue.svg)](https://spacy.io/models/#section-quickstart) [![dep4](https://img.shields.io/badge/nltk-3.4%2B-blue.svg)](https://www.nltk.org/install.html) 

**Indented Usage**: With this script, one could try out different data sets and model parameters to see how the resulting models/chatbots differ.

**Script Summary**:
The goal of the script was to provide an easier way to vary various parameters regarding the seq2seq model (such as encoder & decoder recurrent steps, latent dimensions, training epochs, etc..). Furthermore, the script uses a more memory friendly way of training the seq2seq model, which allows the user to train models with large datasets of question-answer pairs. Also, the script has some caching implemented since "preparing" the training data takes a long time for large datasets. Lastly, the script can save models once it's trained, as well as load trained models to chat with.

## Model Explanation
<p align="center">
  <img src="https://isaacchanghau.github.io/img/nlp/seq2seq-neuralconver/seq2seq.png" width="900">
  <br><i>Sequence to sequence model diagram.</i>
</p>

The underlying model of the ChatBot is a sequence to sequence model which is explained in detail in the following paper: [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf). But at a very high level, it encodes a question (a 'chat' input from the user) using an encoder LSTM to form a thought vector to 'describe' the question as a single vector. Then, the decoder (also a LSTM) takes said thought vector as the initial input and generates a response/answer. The picture above is a good visualization of the model.

**Implementation Details:**
The LSTMs required for the seq2seq model was implemented using [Keras](https://keras.io/) for simplicity. Furthermore, said model uses one-hot encodings of sentences its inputs and outputs. The one hot encodings use a vocab of the most frequent words/tokens of a document. Also, name entity recognition can be used (toggled) in this script and if it is enabled, all questions and answers will effectively have their entities subbed out for their 'entity tokens' before they are one hot encoded. For example, the sentence: "She was born on September 1st, 1996" would effectively be "She was born on <DATE> <DATE> <DATE>" before it is one hot encoded. The NER used in this script is from the [spaCy library](https://spacy.io/). Note that NER considerably increases the time it takes to "prepare" the training data. Lastly, letter casing is ignored in the training data so data strings/sentences are effectively converted to their lower case form (after NER is done). This is done to reduce the vocab size and training data variation. One could do some post-processing on a model's generated response to properly letter case the response (this is not implemented yet). 

## Features
**Veriable Model Parameter:**

* One can change the number of time steps in the encoder and decoder LSTMs as well as change the latent dimensions of said LSTMs. 

* One can define a vocab size (used for the one-hot encoding) as well as the `.json` file used to create the vocab (file format and details are in the [section below](#data--vocab-file-details-and-spec)). 

* One can define the `.json` file used to train the model (file format and details are in the [section below](#data--vocab-file-details-and-spec)).

* One can define the number of epochs used in training as well as the batch size used during training. Note that memory usage largely scales with batch size due to the one-hot encodings. 

* Lastly, since the script uses held out data as its performance measure for the model during training, one can define the percentage of data that is held out for evaluation during training. Note that the held out data is randomly chosen and changes at every epoch.

**Training Memory Efficiency:**

A major advantage of this script is that it does not store all of the training data as one-hot encoded vectors during training. Instead, it stores all of the training data in an intermediate "vocab encoding". The script only creates the one-hot encoding (from said vocab encoding) for a sentence/data point when it's part of a batch that is being trained. Hence why the batch size hugely influences the memory usage. Since an ideal batch size is small, the script doesn't hog a lot of memory (around 1-3 GB of RAM for most 'normal' models). 

> The vocab encoding is as follows: It uses the vocab to create an encoded vector where entry `i` of said vocab encoding is the token id of `L[i]` (`L` is a list of tokens of a given sentence). So for example if a vocab is `{'<PADD>':0, ..., 'how':10, 'are':20, 'you':30, '?':31  ...}` and `L = ["how", "are", "you", "?"]` the resulting vocab encoding would be `[10, 20, 30, 31, 0, ..., 0]` (padded as needed). One can see how easy and efficient it would be to convert this vocab encoding to a one-hot encoding when a batch is needed for training.

**Name Entity Recognition (NER):**

The script has support for NER via the [spaCy library](https://spacy.io/). NER allows us to reduce the vocab size if certain entities appear a lot in the dataset. However more importantly it reduces the variation in the training data, which can result in a loss of granularity in the training data. For example, consider these two sentences: "She was born on September 1st, 1996" and "She was born on October 3rd, 1864". In a more abstract sense, the two sentences are essentially the same sentences as they both describe the birthdate of some girl. If we ignore the entity and treat these two sentences as one, we have less fitting to do for the seq2seq model (or more data to reinforce a correct model). 

Note that currently if NER is enabled, the generated responses/answers will have entity tags instead of actual entities, i.e: the chatbot would generate  "She was born on <DATE> <DATE> <DATE>" instead of the actual sentence mentioned above. One could do some post-processing on the generated response and substitute back appropriate entities (this is not implemented yet).
  
**Data Filtering:**

The script filters the question-answer training data to get more useful q-and-a pairs for the model. As is, the script has 3 different filter modes that can be chosen and they are:

1) Only take Questions that have `N_in` (number of encoder recurrent steps) number of tokens and only take Answers that have `N_out` (number of decoder recurrent steps) number of tokens.

2) All of filter 1 *and* Questions must have a `?` token.

3) All of filter 2 *and* Answers must have a `?` token.

> Filter 1 is so that all training data used fits the defined model. Filter 2 ensures that 'question' are indeed questions. Lastly, filter 3 encourages the model to respond with a question so that the conversation can carry on.

**Vocab and Data Caching:**

The script supports vocab and vocab encoded data caching as those two things can take a long time to generate (especially if NER is enabled). Every time a vocab is created, it is cached, and if the `.json` file for the vocab is the same as the `.json` file used to generate the cache file, the cache vocab is loaded. The vocab encoded data is cached every time it's generated and the cached file is loaded in a case similar to that of the vocab cache file. (Reference the code for details).

**Model Saving:**

Since the goal of the script is to try out various different parameters and datasets, the script can save and load models (vocab, LSTMs and all). The user can choose where to load and save the models. Note that any change to the chatbot object may mess up the saved model, however, there are backups (model weights and vocab pickle files) for each saved model that could be used to reconstruct the model. 

## User Guide
**Dependencies:** Python 3.6+, Numpy, Keras, Tensorflow, nltk, spaCy. It is recommended to have a GPU and have Tensorflow use the GPU (requires a supported NVIDIA GPU). Also, it is recommended to have around 4 GB of system memory for relatively large models with a reasonable batch size. 

### Data & Vocab json file spec
As expected, these files must be a `.json` file and **both** require the following attributes:

* 

### Script options

## Cornell Movie Dialogs Results

## Credits
This script originally started off as a project assignment for [David Bamman's](http://people.ischool.berkeley.edu/~dbamman/) Natural Language Processing Course (Info 159/259) at UC Berkeley (with the goal of just creating a seq2seq chatbot). However the caching, model saving, data filtering, and parameter variations were all added after the assignment was submitted so that one could see how different model variations would perform.
