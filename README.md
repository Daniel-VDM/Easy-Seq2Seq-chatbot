# Seq2Seq Chatbot <div style="text-align: right"> [![dep0](https://img.shields.io/badge/Python-3.6%2B-brightgreen.svg)](https://www.python.org/downloads/) [![dep2](https://img.shields.io/badge/Keras-2.0+-blue.svg)](https://keras.io/) [![dep3](https://img.shields.io/badge/spaCy-2.0%2B-blue.svg)](https://spacy.io/models/#section-quickstart) [![dep4](https://img.shields.io/badge/nltk-3.4%2B-blue.svg)](https://www.nltk.org/install.html) </div>
 
An implementation of a chatbot that uses a sequence to sequence (seq2seq) model which can be trained on any set of question-answer pairs. 

**Indented Usage**: With this script, one could try out different data sets and model parameters to see how the resulting models/chatbots differ.

**Script Summary**:
The goal of the script was to provide an easier way to vary various parameters regarding the seq2seq model (such as encoder & decoder recurrent steps, latent dimensions, training epochs, etc..). Furthermore, the script uses a more memory friendly way of training the seq2seq model, which allows the user to train models with large datasets of question-answer pairs. Also, the script has some caching implemented since "preparing" the training data takes a long time for large datasets. Lastly, the script can save models once it's trained, as well as load trained models to chat with.

## Model Explanation
<p align="center">
  <img src="https://isaacchanghau.github.io/img/nlp/seq2seq-neuralconver/seq2seq.png" width="900">
  <br><i>Sequence to sequence model diagram.</i>
</p>

The underlying model of the ChatBot is a sequence to sequence model which is explained in detail in the following paper: [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf). But at a very high level, it uses an encoder LSTM to encode a question (a 'chat' input from the user) into a *single* thought vector. Then, a seperate decoder  LSTM takes said thought vector as the initial input and generates a response/answer. The picture above is a good visualization of the model.

**Implementation Details:**
The LSTMs required for the seq2seq model was implemented using [Keras](https://keras.io/) for simplicity. Furthermore, said model uses one-hot encodings of sentences its inputs and outputs. The one hot encodings use a vocab of the most frequent words/tokens of a document. Also, name entity recognition can be used (toggled) in this script and if it is enabled, all questions and answers will effectively have their entities subbed out for their 'entity tokens' before they are one hot encoded. For example, the sentence: "She was born on September 1st, 1996" would effectively be "She was born on <DATE> <DATE> <DATE>" before it is one hot encoded. The NER used in this script is from the [spaCy library](https://spacy.io/). Note that NER considerably increases the time it takes to "prepare" the training data. Lastly, letter casing is ignored in the training data so data strings/sentences are effectively converted to their lower case form (after NER is done). This is done to reduce the vocab size and training data variation. One could do some post-processing on a model's generated response to properly capitalize the response (this is not implemented yet). 

## Features
**Veriable Model Parameter:**

* One can change the number of time steps in the encoder and decoder LSTMs as well as change the latent dimensions of said LSTMs. 

* One can define a vocab size (used for the one-hot encoding) as well as the JSON file used to create the vocab (file format and details are in the [section below](#data--vocab-json-file-spec)). 

* One can define the JSON file used to train the model (file format and details are in the [section below](#data--vocab-json-file-spec)).

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

1) Only take Questions that have `n_in` (number of encoder recurrent steps) number of tokens and only take Answers that have `n_out` (number of decoder recurrent steps) number of tokens.

2) All of filter 1 *and* Questions must have a `?` token.

3) All of filter 2 *and* Answers must have a `?` token.

> Filter 1 is so that all training data used fits the defined model. Filter 2 ensures that 'question' are indeed questions. Lastly, filter 3 encourages the model to respond with a question so that the conversation can carry on.

**Vocab and Data Caching:**

The script supports vocab and vocab encoded data caching as those two things can take a long time to generate (especially if NER is enabled). Every time a vocab is created, it is cached, and if the JSON file for the vocab is the same as the JSON file used to generate the cache file, the cache vocab is loaded. The vocab encoded data is cached every time it's generated and the cached file is loaded in a case similar to that of the vocab cache file. (Reference the code for details).

**Training Model Recovery:**

This script saves a 'recovery' model in the cache at the end of each epoch during training. If the script were to get interrupted for whatever reason, it can recover its training progress by loading said model and resume training. 

**Model Saving:**

Since the goal of the script is to try out various different parameters and datasets, the script can save and load models (vocab, LSTMs and all). The user can choose where to load and save the models. Note that any change to the chatbot script may mess up the saved model, however, there are backups (model weights and vocab pickle files) for each saved model that could be used to reconstruct the model. 

## User Guide
**Dependencies:** Python 3.6+, Numpy, Keras, Tensorflow, nltk, spaCy. It is recommended to have Tensorflow (or whatever Keras backend) work with a GPU since large models will take a considerable amount of time to train (this will require a supported NVIDIA GPU). Also, it is recommended to have around 4 GB of system memory for relatively large models with a reasonable batch size. 

### Data & Vocab JSON file spec
The Data and Vocab file must be a JSON file and **both** have the following attributes:

* Attribute: "data". For the data file this can be a list of question-answer pairs, i.e: `[...,["Did you change your hair?", "No."], ["Hi!", "Hello."],...]`. For the vocab file, this can be just a list of sentences *or* a list of question-answer pairs.

* Attribute: "questions". Optional for the vocab file but mandatory for the data file. This is simply the list of questions from the question-answer pairs (for convenience). 

* Attribute: "answers". Optional for the vocab file but mandatory for the data file. This is simply the list of answers from the question-answer pairs (for convenience). 

* Attribute: "signature". Mandatory for both. This is some sort of (string) identifier that ties back to the original source of the data, i.e: source_file_name + last modified time of source_file_name.

> Sample JSON files can be found with the script ([`Cornell_Movie_Dialogs_Data.json`](Cornell_Movie_Dialogs_Data.json) & [`Small_Data.json`](Small_Data.json)). Furthermore, one could reference [`./training_data_scripts/Cornell-Data_json_creator.py`](training_data_scripts/Cornell-Data_json_creator.py) as a sample script that takes a CSV file and creates the desired JSON file.

### Script options
The script has various options that are handled by an options parser. To look up the options and their quick descriptions use the `--help` option, i.e: use the command: `python chatbot.py --help`.

Here is the help message for reference:
```
usage: chatbot.py [-h] [-i N_IN] [-o N_OUT] [-l LATENT_DIM] [-v VOCAB_SIZE]
                  [-f VOCAB_FILE] [-I] [-N] [-M] [-t TRAIN_FILE]
                  [-c FILTER_MODE] [-e EPOCH] [-b BATCH_SIZE] [-s SPLIT]
                  [-m SAVED_MODELS_DIR]

optional arguments:
  -h, --help            show this help message and exit
  -i N_IN, --n_in N_IN  The number of time steps for the encoder. Default =
                        10.
  -o N_OUT, --n_out N_OUT
                        The number of time setps for the decoder. Default =
                        20.
  -l LATENT_DIM, --latent_dim LATENT_DIM
                        The inner dimensionality of the Encoder and Decoder's
                        LSTM. Default = 128.
  -v VOCAB_SIZE, --vocab_size VOCAB_SIZE
                        The size of the vocab of the Chatbot. Default = None
  -f VOCAB_FILE, --vocab_file_path VOCAB_FILE
                        The directory of the JSON file that is used to define
                        the vocab. The 'data' attribute can be either
                        question-answer pairs or just strings/sentences.
                        Default = whatever the train_file is.
  -I, --ignore_cache    Forces the script to ignore the cached files.
  -N, --NER_disable     Turns off Name Entity Recognition as part of the
                        chatbot model. Note that NER adds a considerable
                        amount of complexity in encoding the training data.
  -M, --verbose         Toggles verbose on.
  -t TRAIN_FILE, --train_file_path TRAIN_FILE
                        The directory of the JSON file that is used to train
                        the model. The 'data' attribute must be a list of
                        question-answer pairs.Default =
                        'Cornell_Movie_Dialogs_Data.json'
  -c FILTER_MODE, --filter_mode FILTER_MODE
                        An integer that dictates the filter imposed of the
                        data. MODES: {0, 1, 2}. Mode 0: Only take Questions
                        that have N_in number of tokens and only take Answers
                        that have n_out number of tokens. Mode 1: All of Mode
                        0 AND Questions must have a '?' token. Mode 2: All of
                        Mode 0 AND Question & Answer must have a '?' token.
                        Default = 0
  -e EPOCH, --epoch EPOCH
                        The number of epochs for training. Default = 100.
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        The batch size for training. Default = 32.
  -s SPLIT, --split SPLIT
                        The percentage (float between 0 - 1) of data held out
                        for validation. Default = 0.35
  -m SAVED_MODELS_DIR, --saved_models_dir SAVED_MODELS_DIR
                        The directory for all of the saved (trained) models.
                        Default = 'saved_models'
```

## Sample Execution
One could run the script with the following command: 

`python chatbot.py --n_in=20 --n_out=20 --latent_dim=128 --vocab_size=10000 --train_file_path=Small_Data.json --filter_mode=1 --epoch=600 --batch_size=32 --split=0.35 --verbose`

**When training, one should get something similar to the following: (if cached files are invalid)**
```
Daniels-MacBook-Pro:Seq2Seq-chatbot danielvdm$ python chatbot.py --n_in=10 --n_out=20 --latent_dim=128 --vocab_size=10000 --train_file_path=Small_Data.json --filter_mode=1 --epoch=500 --batch_size=64 --split=0.35 --verbose
Using TensorFlow backend.
Load a saved model? (y/n) n
Save the newly trained model? (y/n) y
(Required) Name of newly trained model? Example_Model
No cached vocab file found.
Exception encountered when loading vocab assets: FileNotFoundError, [Errno 2] No such file or directory: 'cache/vocab_assets/Small_Data.json (last_mod: 1547701140.2666988).pickle'
>> Creating vocab assets for 'Small_Data.json' (95 lines of data) <<
Parsed 95/95 lines of vocab data.Cached vocab file. Vocab size = 10000, Vocab Sig = Small_Data.json (last_mod: 1547701140.2666988)
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            (None, None, 232)    0
__________________________________________________________________________________________________
input_2 (InputLayer)            (None, None, 232)    0
__________________________________________________________________________________________________
lstm_1 (LSTM)                   [(None, 128), (None, 184832      input_1[0][0]
__________________________________________________________________________________________________
lstm_2 (LSTM)                   [(None, None, 128),  184832      input_2[0][0]
                                                                 lstm_1[0][1]
                                                                 lstm_1[0][2]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, None, 232)    29928       lstm_2[0][0]
==================================================================================================
Total params: 399,592
Trainable params: 399,592
Non-trainable params: 0
__________________________________________________________________________________________________
None
Defined new model.

>> Vocab Encoding Training Data <<
Processed 95/95 Question-Answer Pairs.
Cached vocab encoded training data.
>> Training on 41 Question-Answer pairs <<
Epoch: 1/500, Batch: 1/1. 	Training...
Train on 27 samples, validate on 14 samples
Epoch 1/1
27/27 [==============================] - 4s 142ms/step - loss: 5.4266 - val_loss: 5.3929
Epoch: 2/500, Batch: 1/1. 	Training...
Train on 27 samples, validate on 14 samples
Epoch 1/1
27/27 [==============================] - 0s 5ms/step - loss: 5.3927 - val_loss: 5.3518
. 
. 
.
Epoch: 499/500, Batch: 1/1.     Training...
Train on 27 samples, validate on 14 samples
Epoch 1/1
27/27 [==============================] - 0s 5ms/step - loss: 0.2527 - val_loss: 0.2114
Epoch: 500/500, Batch: 1/1.     Training...
Train on 27 samples, validate on 14 samples
Epoch 1/1
27/27 [==============================] - 0s 5ms/step - loss: 0.2215 - val_loss: 0.2687
Finished epoch: 500/500
Training Complete.
Trained on 41 Question-Answer pairs

Saved the trained model to: './saved_models/Example_Model'.
Chat-bot ready, type anything to start: (Ctrl + C or type '!EXIT' to stop chatting)
>Hi?
Response: how are you ?

>^C
Done Chatting...
```

**When loading a model, one should get something similar to the following: (if more than 1 model is saved)**
```
Daniels-MacBook-Pro:Seq2Seq-chatbot danielvdm$ python chatbot.py
Using TensorFlow backend.
Load a saved model? (y/n) y
Which model would you like to load? (Type out choice below)
  List of saved models:
        ['Test', 'Test2', 'Example_Model']
>Example_Model

Loaded model: Example_Model
Chat-bot ready, type anything to start: (Ctrl + C or type '!EXIT' to stop chatting)
>Hi?
Response: how are you ?

>^C
Done Chatting...
```

## Movie Script Results
**Data info:**

The 'large' model saved in this repo was trained on [Cornell's Movie Dialogs dataset](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html). Said dataset came as a CSV file with 5 columns: LineID, characterID, movieID, character name and text. When converting the CSV file to the JSON file for the script, the following filters were applied: First, it only considers rows where the text is 20 tokens or less. Next, for every row `i` that passes said first filter, the text of row `i` was a question and text of row `i+1` was the respective answer so long as the movieIDs were the same, characterIDs were different and LineIDs were consecutive.

The 'small' model saved in this repo was trained on some handwritten [test data](training_data_scripts/dataset.txt). Each consecutive string in said dataset was a question-answer pair in the JSON file used for the vocab and training.

**Training info:**

The 'large' model parameters that yielded the best result were ... **TBD, still training**

The 'small' model parameters that yeilded the best results were: `n_in = 20, n_out = 20, Latent_Dim = 128, Vocab_Size = None, Epoch = 600, Batch_Size = 32, Split = 0.35, Filter_Mode = 1`. The model ended up with a held out loss of approximatly 0.05.

**Sample Conversations:**

*'Large' Model*

*'Small' Model*


## Credits
This project was written by Daniel Van Der Maden as an Undergraduate Computer Science student at UC Berkeley.

This script originally started off as a project assignment for [David Bamman's](http://people.ischool.berkeley.edu/~dbamman/) Natural Language Processing Course (Info 159/259 - Fa18) at UC Berkeley (with the goal of just creating a seq2seq chatbot). However, memory efficiency changes, caching, model recovering & saving, data filtering, and parameter variations were all added at a later time so that one could more easily see how different model variations would perform.
