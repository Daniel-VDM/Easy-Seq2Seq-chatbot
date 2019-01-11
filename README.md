# Seq2Seq-chatbot
An implementation of a chatbot that uses a sequence to sequence (seq2seq) model which can be trained on any set of question-answer pairs. 

**Indented Usage**: With this script, one could try out different data sets and model parameters to see how the resulting models/chatbots differ.

**Script Summary**:
The goal of the script was to provide an easier way to very various parameters regarding the seq2seq model (such as encoder & decoder recurrent steps, latent dimensions, training epochs, etc..). Furthermore, the script uses a more memory friendly way of training the seq2seq model, which allows the user to train the model with large data sets of question-answer pairs. Also, the script has some caching implemented since "preparing" the training data takes a long time for large datasets. Lastly, the script can save models once it's trained, as well as load trained models to chat with.

## Model Explanation
<p align="center">
  <img src="https://isaacchanghau.github.io/img/nlp/seq2seq-neuralconver/seq2seq.png" width="700">
  <br><i>Sequence to sequence model diagram.</i>
</p>

## Features

## User Guide

## Cornell Movie Dialogs Results