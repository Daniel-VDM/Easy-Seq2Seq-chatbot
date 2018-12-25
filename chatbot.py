import numpy as np
import nltk
import os
import sys
import pickle
import itertools
import spacy
import shutil
import json
from optparse import OptionParser
from collections import deque
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.preprocessing import sequence

NLP = spacy.load('en')


def define_models(n_input, n_output, n_units):
    """
    Seq2Seq encoder/decoder definition function.
    """
    # define training encoder
    encoder_inputs = Input(shape=(None, n_input))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    # define training decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    # return all models
    return model, encoder_model, decoder_model


class ChatBot:

    def __init__(self, n_in, n_out, vocab_size, vocab_file, ignore_cached_vocab=False, ner_enabled=True):
        self.ner_enabled = ner_enabled

        if os.path.isfile("cached_vocab.pickle") and not ignore_cached_vocab:
            try:
                data_vocab_dicts = pickle.load(open("cached_vocab.pickle", 'rb'))
                if len(data_vocab_dicts["word_to_id"]) != len(data_vocab_dicts["word_to_id"]):
                    raise ValueError("'cached_vocab.pickle' dictionary lengths do not match.")
                if len(data_vocab_dicts["word_to_id"]) != vocab_size:
                    raise ValueError("'cached_vocab.pickle' vocab size is not {}.".format(vocab_size))
                if self.ner_enabled and ("NER_tokens" not in data_vocab_dicts.keys()
                                         or "NER_label_to_token_dict" not in data_vocab_dicts.keys()):
                    raise ValueError("'cached_vocab.pickle' does not contain NER data.")
            except Exception as e:
                print("Exception encountered when reading vocab data: {}".format(e))
                data_vocab_dicts = self._create_and_cache_vocab(vocab_file, vocab_size, ner_enabled)
        else:
            data_vocab_dicts = self._create_and_cache_vocab(vocab_file, vocab_size, ner_enabled)

        self.n_in, self.n_out = n_in, n_out
        self.vocab_size = vocab_size
        self.word_to_id_dict = data_vocab_dicts["word_to_id"]
        self.id_to_word_dict = data_vocab_dicts["id_to_word"]
        self.encoder, self.decoder = None, None

        if self.ner_enabled:
            self.ner_tokens = data_vocab_dicts["NER_tokens"]
            self.ner_label_to_token_dict = data_vocab_dicts["NER_label_to_token_dict"]

    def __del__(self):
        # TODO: delete temp files...
        pass

    def __bool__(self):
        return self.encoder is not None and self.decoder is not None

    @staticmethod
    def _create_and_cache_vocab(vocab_file, vocab_size, ner_enabled=True):
        """
        Private Static Method.

        Creates and pickle's vocab from VOCAB_FILE. Note that VOCAB_FILE
        is expected to come as a json file where said file is a list of
        question-answer pairs:
            For example:
                [...,["Did you change your hair?", "No."], ["Hi!", "Hello."],...]
        Note that said file needs to be in the same dir as this script.

        Vocab uses most frequent words first when truncating the vocab to
        fit the vocab size.

        Note that the cached vocab also saves a set of NER tokens (from the given
        vocab file) for future references.

        This function is very expensive due to the NER tagging.

        TO SELF: This should be improved in the future to incorperate a better NER tagger
        that takes advantage for the Cornell DB structure. I.E NLP the whole movie and
        tag from there...

        :param vocab_file: file name of the jason vocab file used to generate the vocab.
        :param vocab_size: the fixed size of the vocab.
        :param ner_enabled: toggles NER encoding for vocab.
        """
        word_freq, i = {}, 0
        vocab_data = json.load(open(vocab_file))
        ner_label_tokens = set()
        ner_tokens = set()
        ner_label_to_token_dict = {}

        for question, answer in vocab_data:
            if ner_enabled:
                for entity in itertools.chain(NLP(question).ents, NLP(answer).ents):
                    ner_label_tokens.add(f"<{entity.label_}>")
                    for wrd in nltk.word_tokenize(entity.text):
                        ner_tokens.add(wrd)
                        if f"<{entity.label_}>" in ner_label_to_token_dict:
                            ner_label_to_token_dict[f"<{entity.label_}>"].add(wrd)
                        else:
                            ner_label_to_token_dict[f"<{entity.label_}>"] = {wrd}

            for tok in itertools.chain(nltk.word_tokenize(question), nltk.word_tokenize(answer)):
                if tok.isalpha() and tok not in ner_tokens:
                    tok = tok.lower()
                    if tok in word_freq:
                        word_freq[tok] += 1
                    else:
                        word_freq[tok] = 1

            i += 1
            sys.stdout.write(f"\rCreating Vocab, parsing {i}/{len(vocab_data)} question-answer pairs.")
            sys.stdout.flush()

        special_tokens = ["<PADD>", "<START>", "<UNK>"] + list(ner_label_tokens)

        vocab = sorted(list(word_freq.keys()), key=lambda w: word_freq.get(w, 0), reverse=True)
        vocab = vocab[:vocab_size - len(special_tokens)]

        if ner_enabled:
            dump = {"word_to_id": {c: i for i, c in enumerate(itertools.chain(special_tokens, vocab))},
                    "id_to_word": {i: c for i, c in enumerate(itertools.chain(special_tokens, vocab))},
                    "NER_tokens": ner_tokens, "NER_label_to_token_dict": ner_label_to_token_dict}
        else:
            dump = {"word_to_id": {c: i for i, c in enumerate(itertools.chain(special_tokens, vocab))},
                    "id_to_word": {i: c for i, c in enumerate(itertools.chain(special_tokens, vocab))}}
        pickle.dump(dump, open("cached_vocab.pickle", 'wb'))
        print("\nCached vocab file.")
        return dump

    @staticmethod
    def _list_one_hot_encode(iterable, vocab_len):
        """
        Private method for decoder.

        TODO: refactor decoder to not use this private method.
        """
        lst = []
        for seq in iterable:
            time_steps = []
            for index in seq:
                vector = [0 for _ in range(vocab_len)]
                vector[index] = 1
                time_steps.append(vector)
            lst.append(time_steps)
        return lst

    @staticmethod
    def _array_one_hot_encode(iterable, n, vocab_len):
        """
        Private method for train method.

        TODO: DOCS
        """
        encoding = np.zeros((len(iterable), n, vocab_len))
        for i, seq in enumerate(iterable):
            for j, index in enumerate(seq):
                encoding[i, j, index] = 1
        return encoding

    @staticmethod
    def _create_validation_split(X_1, X_2, Y, percentage):
        """
        Private method for train method.

        TODO: DOCS
        """
        X_1t, X_2t, Y_t = [X_1[1]], [X_2[1]], [Y[1]]
        X_1v, X_2v, Y_v = [X_1[0]], [X_2[0]], [Y[0]]
        for i in range(2, X_1.shape[0]):
            if np.random.uniform() < percentage:
                X_1v.append(X_1[i])
                X_2v.append(X_2[i])
                Y_v.append(Y[i])
            else:
                X_1t.append(X_1[i])
                X_2t.append(X_2[i])
                Y_t.append(Y[i])
        X_1t, X_2t, Y_t = np.array(X_1t), np.array(X_2t), np.array(Y_t)
        X_1v, X_2v, Y_v = np.array(X_1v), np.array(X_2v), np.array(Y_v)
        return X_1t, X_2t, Y_t, X_1v, X_2v, Y_v

    def vectorize(self, sentence, length=None):
        """
        Note that this is NOT one-hot encoded. Instead, it returns a vector where
        each entry is a word ID, and said entry corresponds to token index of sentence.

        :param sentence: A string that is to be vectorized.
                         Note that it CAN include punctuation and unknown words.
        :param length: The length of the returned vector. Note that it defaults to
                       the number of tokens in SENTENCE.
        :return: an encoding/vector (using this objects vocab) of the sentence.
        """
        sentence = sentence.strip()
        unk_token_id = self.word_to_id_dict["<UNK>"]
        sentence_tokens = list(filter(lambda s: s.isalpha(), nltk.word_tokenize(sentence)))
        length = length if length else len(sentence_tokens)
        vector = np.zeros(length, dtype=int)

        entity = {}
        if self.ner_enabled and any(w for w in sentence_tokens if w in self.ner_tokens):
            for ent in NLP(sentence).ents:
                for w in nltk.word_tokenize(ent.text):
                    entity[w] = f"<{ent.label_}>"

        for i, word in zip(range(length), sentence_tokens):
            if word in entity:
                word_id = self.word_to_id_dict[entity[word]]
            else:
                word_id = self.word_to_id_dict.get(word.lower(), unk_token_id)
            vector[i] = word_id
        return vector

    def _encode_and_store_training_data(self, data_file, temp_store_dir='temp', verbose=0):
        """
        Private method to create and save an 'encoded' version of the training data
        (from DATA_FILE) to the STORE_DIR directory.

        The 'encoding' is as follows:
            Given a sentence, we create an array/vector of size N_in or N_out
            (depending on which data we are encoding) where element i of
            said vector is token i's word ID in the word_to_id_dict dictionary
            of this object.

        Note each question in the training data gets its own vector & file and
        each answer gets 2 vectors & files (one of them is shifted by 1 time step).

        :param data_file: The data file from the 'train' method. ()
        :param temp_store_dir: The directory in which the encoding is stored in.
        :param verbose: update messages during execution.
        :return: The number question-answer pairs encoded.
        """

        def is_valid_data(question, answer):
            q_toks = [tok for tok in nltk.word_tokenize(question) if tok.isalpha()]
            a_toks = [tok for tok in nltk.word_tokenize(answer) if tok.isalpha()]
            return len(q_toks) <= self.n_in and len(a_toks) <= self.n_out

        count = 0
        # directory = f"{temp_store_dir}/{id(self)}"
        directory = f"{temp_store_dir}/DEV"
        data = json.load(open(data_file))

        if not os.path.exists(directory):
            os.makedirs(directory)

        for i, (q, a) in enumerate(data):
            if is_valid_data(q, a):
                q_vec = self.vectorize(q, self.n_in)
                a_vec = self.vectorize(a, self.n_out)
                a_shift_vec = np.roll(a_vec, 1)
                a_shift_vec[0] = self.word_to_id_dict["<START>"]
                pickle.dump(q_vec, open(f"{directory}/{count}_x1.pickle", 'wb'))
                pickle.dump(a_vec, open(f"{directory}/{count}_y.pickle", 'wb'))
                pickle.dump(a_shift_vec, open(f"{directory}/{count}_x2.pickle", 'wb'))
                count += 1
            if verbose:
                sys.stdout.write(f"\rVectorizing Training data {i}/{len(data)}")
                sys.stdout.flush()

        return count

    def _training_vectors_generator(self, batch_size=32, temp_store_dir='temp', verbose=0):
        """Private Generator for the 'train' method of this object.

        Note that this uses the saved 'encoding' in STORE_DIR (generated by the
        '_encode_and_store_training_data' method) to create the one-hot matrices
        used for training. Said encodings are identified by this object's ID in
        STORE_DIR.

        :param batch_size: The size of the batch used in training.
        :param temp_store_dir: he directory in which the encoding is stored in.
        :param verbose: update messages during execution.
        :return: A generator.
        """
        directory = f"{temp_store_dir}/DEV"
        # TODO: THIS SHIT NEXT...
        pass

    def _create_training_generator(self, data_file, sentence_limit, batch_size=32):
        """
        A generator that yields batches of one-hot encoded sentences for
        the Seq2Seq model.

        DATA_FILE is expected to come from 'self.train' method's DATA_FILE argument.

        Also, one-hot encoding comes from the vocab's dictionaries.

        :param data_file: The data file from the 'train' method.
        :param sentence_limit: The sentence limit for all sentences in the training data.
        :param batch_size: The size of the batch used in training.
        """
        q_and_a_lst = []

        data = open(data_file, encoding='utf-8', errors='ignore').read().split('\n')
        if not data[-1]:
            data.pop()

        for i in range(len(data) - 1):
            line_a, a_uter, a_mov, _, a_text = data[i].split("\t")[:5]
            line_b, b_uter, b_mov, _, b_text = data[i + 1].split("\t")[:5]
            line_a = int("".join([s for s in line_a if s.isdigit()]))
            line_b = int("".join([s for s in line_b if s.isdigit()]))

            if a_uter != b_uter and a_mov == b_mov and line_b == line_a + 1 \
                    and len(a_text.split(" ")) <= sentence_limit \
                    and len(b_text.split(" ")) <= sentence_limit:
                q_and_a_lst.append((a_text.strip(), b_text.strip()))

        batch_number, doc_number = 0, 0
        total_batch_count = int(np.ceil(len(q_and_a_lst) / batch_size))
        np.random.shuffle(q_and_a_lst)
        queue = deque(q_and_a_lst)

        while queue:
            batch_size = min(len(queue), batch_size)
            lst = [queue.pop() for _ in range(batch_size)]

            questions = map(lambda tup: tup[0], lst)
            answers = map(lambda tup: tup[1], lst)

            X_1 = np.empty((batch_size,), dtype=bytearray)
            X_2 = np.empty((batch_size,), dtype=bytearray)
            Y = np.empty((batch_size,), dtype=bytearray)

            # Create training vectors from read data.
            for index, question in enumerate(questions):
                X_1[index] = self.vectorize(question)

                doc_number += 1
                sys.stdout.write("\rVectorizing Training data {}/{}".format(doc_number, 2 * len(q_and_a_lst)))
                sys.stdout.flush()

            for index, answer in enumerate(answers):
                vector = self.vectorize(answer)
                Y[index] = vector
                vector = [self.word_to_id_dict["<START>"]] + list(vector)[:-1]
                X_2[index] = np.array(vector)

                doc_number += 1
                sys.stdout.write("\rVectorizing Training data {}/{}".format(doc_number, 2 * len(q_and_a_lst)))
                sys.stdout.flush()

            X_1 = sequence.pad_sequences(X_1, maxlen=self.n_in, padding='post')
            X_2 = sequence.pad_sequences(X_2, maxlen=self.n_out, padding='post')
            Y = sequence.pad_sequences(Y, maxlen=self.n_out, padding='post')

            encode_len = len(self.word_to_id_dict)
            X_1 = self._array_one_hot_encode(X_1, self.n_in, encode_len)
            X_2 = self._array_one_hot_encode(X_2, self.n_out, encode_len)
            Y = self._array_one_hot_encode(Y, self.n_out, encode_len)

            batch_number += 1

            yield X_1, X_2, Y, "{}/{}".format(batch_number, total_batch_count)

    def train(self, data_file, epoch, temp_store_dir='temp', batch_size=32, split_percentage=0.35, verbose=0):
        """
        Trains the chatbot's encoder and decoder LSTMs (= the Seq2Seq model).

        Note that DATA_FILE is expected to come as a json file where said
        file is a list of question-answer pairs:
            For example:
                [...,["Did you change your hair?", "No."], ["Hi!", "Hello."],...]
        Note that said file needs to be in the same dir as this script.

        :param data_file: the data being train on. Must follow format above.
        :param epoch: number of epochs in training.
        :param batch_size: size of the batch in training.
        :param split_percentage: a float between 0 and 1. It is the percentage of
                                 training data held out for validation.
        :param verbose: update messages during training.
        """
        model, encoder, decoder = define_models(len(self.word_to_id_dict), len(self.id_to_word_dict), 128)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        if verbose:
            print(model.summary())

        print("-==Preparing Training Data==-")
        self._encode_and_store_training_data(data_file=data_file, temp_store_dir=temp_store_dir, verbose=verbose)
        sys.exit(1)

        print("\n\n-==TRAINING=--\n")
        for ep in range(epoch):
            batch_gen = self._create_training_generator(data_file=data_file,
                                                        batch_size=batch_size,
                                                        sentence_limit=20)
            for X_1, X_2, Y, batch_counter in batch_gen:
                if verbose:
                    sys.stdout.flush()
                    sys.stdout.write('\x1b[2K')
                    print("\rEpoch: {}/{}, Batch: {}. \tTraining...".format(ep, epoch, batch_counter))
                X_1t, X_2t, Y_t, X_1v, X_2v, Y_v = self._create_validation_split(X_1, X_2, Y, split_percentage)
                model.fit([X_1t, X_2t], Y_t, epochs=1, batch_size=batch_size,
                          validation_data=([X_1v, X_2v], Y_v), verbose=verbose)
        self.encoder = encoder
        self.decoder = decoder

    def _predict(self, X_in):
        """ Private method used for main chat loop.
        """
        curr_in_state = self.encoder.predict(X_in)
        curr_out_state = [
            np.array(self._list_one_hot_encode([[self.word_to_id_dict["<START>"]]], len(self.word_to_id_dict)))
        ]
        Y_hat = []
        for t in range(self.n_in):
            prediction, h, c = self.decoder.predict(curr_out_state + curr_in_state)
            curr_in_state = [h, c]
            curr_out_state = [prediction]
            Y_hat.append(prediction)
        return np.array(Y_hat)

    def _vector_to_words(self, vector):
        """ Private method used for main chat loop.
        """
        words = []
        for el in vector:
            word_id = np.argmax(el)  # Fetch index that has 1 as element.
            word = self.id_to_word_dict.get(word_id, "<UNK>")
            if word == "<PADD>":
                return " ".join(words)
            words.append(word)
        return " ".join(words)

    def chat(self):
        """ Main chat loop with the chatbot.
        """
        if not self:
            raise RuntimeError("Attempted to chat with an untrained model.")
        print("Chat Bot ready, type anything to start:")
        while True:
            sys.stdout.write(">")
            sys.stdout.flush()
            input_str = input()
            # TODO: refactor the vectorize to not use sequences pad
            vocab_encoded_X_in = sequence.pad_sequences([self.vectorize(input_str)],
                                                        maxlen=self.n_in, padding='post')
            X_in = np.array(self._list_one_hot_encode(vocab_encoded_X_in, len(self.word_to_id_dict)))
            Y_hat = self._predict(X_in)
            print("Response: {}".format(self._vector_to_words(Y_hat)))
            print(" ")


def get_options():
    opts = OptionParser()
    opts.add_option('-i', '--N_in', dest='N_in', type=int, default=10,
                    help="The number of time steps for the encoder. Default = 10.")
    opts.add_option('-o', '--N_out', dest='N_out', type=int, default=20,
                    help="The number of time setps for the decoder. Default = 20.")
    opts.add_option('-v', '--vocab_size', dest='vocab_size', type=int, default=10000,
                    help='The size of the vocab of the Chatbot. Default = 10000')
    opts.add_option('-f', '--vocab_file', dest='vocab_file', type=str, default="Cornell_Movie_Dialogs_Data.json",
                    help="The directory of the file that is used to define the vocab. "
                         "This file must be a json file that contains a list of question-answer"
                         "pairs/lists. Reference the included/default file for details."
                         "Default = 'Cornell_Movie_Dialogs_Data.json'")
    opts.add_option("-I", '--ignore_cached_vocab', action="store_true", dest="ignore_cached_vocab",
                    help="Forces the script to ignore the cached vocab file. "
                         "Thus creating a new vocab file (and cached vocab file) for training.")
    opts.add_option("-N", '--NER', action="store_true", dest="NER_enabled",
                    help="Toggles the use of Name Entity Recognition as part of the chatbot model. "
                         "Note that NER adds a considerable amount of complexity in encoding"
                         "the training data.")
    opts.add_option("-M", '--verbose', action="store_true", dest="verbose",
                    help="Toggles verbose on.")
    opts.add_option('-d', '--data_file', dest='data_file', type=str, default="Cornell_Movie_Dialogs_Data.json",
                    help="The directory of the file that is used to train the model. "
                         "This file must be a json file that contains a list of question-answer"
                         "pairs/lists. Reference the included/default file for details."
                         "Default = 'Cornell_Movie_Dialogs_Data.json'")
    opts.add_option('-e', '--epoch', dest='epoch', type=int, default=500,
                    help="The number of epochs for training. Default = 100.")
    opts.add_option('-b', '--batch_size', dest='batch_size', type=int, default=32,
                    help="The batch size for training. Default = 32.")
    opts.add_option('-s', '--split', dest='split', type=float, default=0.35,
                    help="The percentage (val between 0 - 1) of data held out for validation. "
                         "Default = 0.35")
    opts.add_option('-t', '--temp_store_dir', dest='temp_store_dir', type=str, default="temp",
                    help="The directory used to store all temporary files. Note that this "
                         "directory is heavily used during training so its recommended to use/make "
                         "a RAM disk for this directory. Default = 'temp'.")
    opts.add_option('-m', '--saved_models_dir', dest='saved_models_dir', type=str, default="saved_models",
                    help="The directory for all of the saved (trained) models. "
                         "Default = 'saved_models'")
    return opts.parse_args()[0]


def get_saved_model_dir():
    saved_models = os.listdir("saved_models")
    if len(saved_models) == 0:
        print("There are no saved models.")
        return None
    if len(saved_models) == 1:
        return saved_models[0]
    print("Which model would you like to load? (Type out choice below)")
    print(f"  List of saved models:\n\t{saved_models}")
    saved_models_set = set(saved_models)
    while True:
        sys.stdout.write("\r>")
        sys.stdout.flush()
        choice = input()
        if choice in saved_models_set:
            return choice
        print(f"'{choice}' is invalid. Choose a valid model from the list of saved models.")


if __name__ == "__main__":
    # TODO: IMPLEMENTED AND TEST DECODER NER FEATURES.
    # TODO: ALSO TEST MODELS THAT DO NOT HAVE NERs.
    # TODO: refactor for efficientcy and clenlyness and names of files...
    # TODO: Documentation & REFACTOR TO REMOVE REDUNDANT INFO.
    # TODO: First Time installer...
    # TODO: publish README...
    opts = get_options()
    if not os.path.exists(opts.temp_store_dir):
        os.makedirs(opts.temp_store_dir)
    if not os.path.exists(opts.saved_models_dir):
        os.makedirs(opts.saved_models_dir)

    sys.stdout.write("\rLoad a saved model? (y/n) ")
    sys.stdout.flush()
    user_input = input()

    saved_model_dir = get_saved_model_dir() if user_input[0].lower() == 'y' else None

    if saved_model_dir is not None:
        chat_bot = pickle.load(open(f"{opts.saved_models_dir}/{saved_model_dir}/chatbot.pickle", 'rb'))
        print(f"\nLoaded model: {saved_model_dir}")
    else:
        new_model_name = None

        sys.stdout.write("\rSave the newly trained model? (y/n) ")
        sys.stdout.flush()
        user_input = input()

        if user_input[0].lower() == 'y':
            sys.stdout.write("\r(Required) Name of newly trained model? ")
            sys.stdout.flush()
            new_model_name = input()

        chat_bot = ChatBot(opts.N_in, opts.N_out, opts.vocab_size, opts.vocab_file,
                           opts.ignore_cached_vocab, opts.NER_enabled)
        chat_bot.train(opts.data_file, opts.epoch, opts.temp_store_dir, opts.batch_size, opts.split, opts.verbose)

        if new_model_name:
            new_model_path = f"{opts.saved_models_dir}/{new_model_name}"
            if not os.path.exists(new_model_path):
                os.mkdir(new_model_path)
            pickle.dump(chat_bot, open(f"{new_model_path}/chatbot.pickle", 'wb'))
            if not os.path.exists(f"{new_model_path}/backup"):
                os.mkdir(f"{new_model_path}/backup")
            chat_bot.encoder.save_weights(f"{new_model_path}/backup/encoder.h5")
            chat_bot.decoder.save_weights(f"{new_model_path}/backup/decoder.h5")
            shutil.copyfile("cached_vocab.pickle", f"{new_model_path}/backup/cached_vocab.pickle")
            shutil.copyfile(opts.data_file, f"{new_model_path}/backup/[TRAIN_DATA]{opts.data_file}")
            shutil.copyfile(opts.vocab_file, f"{new_model_path}/backup/[VOCAB_DATA]{opts.vocab_file}")
            print(f"\nSaved the trained model to: '{new_model_path}'.")
    chat_bot.chat()
