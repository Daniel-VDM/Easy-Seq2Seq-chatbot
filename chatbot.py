import numpy as np
import os
import sys
import pickle
import itertools
import spacy
import shutil
import json
import re
import nltk
from collections import deque
from optparse import OptionParser
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

    # TODO: major refactoring of logic to make it presentable.

    def __init__(self, n_in, n_out, vocab_size, vocab_file, ignore_cache=False, ner_enabled=True):
        if not os.path.exists("cache"):
            os.makedirs("cache")

        self.ner_enabled = ner_enabled
        self.ignore_cache = ignore_cache

        if os.path.isfile("cache/vocab.pickle") and not ignore_cache:
            try:
                data_vocab_dicts = pickle.load(open("cache/vocab.pickle", 'rb'))
                if len(data_vocab_dicts["word_to_id"]) != len(data_vocab_dicts["word_to_id"]):
                    raise ValueError("'cache/vocab.pickle' dictionary lengths do not match.")
                if len(data_vocab_dicts["word_to_id"]) != vocab_size:
                    raise ValueError("'cache/vocab.pickle' vocab size is not {}.".format(vocab_size))
                if self.ner_enabled and ("NER_tokens" not in data_vocab_dicts.keys()
                                         or "NER_label_to_token_dict" not in data_vocab_dicts.keys()):
                    raise ValueError("'cache/vocab.pickle' does not contain NER data.")
                print("[!] Using cached vocab.")
            except Exception as e:
                print("Exception encountered when reading vocab data: {}".format(e))
                data_vocab_dicts = self._create_and_cache_vocab(vocab_file, vocab_size, ner_enabled)
        else:
            print("No cached vocab file found.")
            data_vocab_dicts = self._create_and_cache_vocab(vocab_file, vocab_size, ner_enabled)

        self.vocab_file = vocab_file
        self.n_in, self.n_out = n_in, n_out
        self.vocab_size = vocab_size
        self.word_to_id_dict = data_vocab_dicts["word_to_id"]
        self.id_to_word_dict = data_vocab_dicts["id_to_word"]
        self.encoder, self.decoder = None, None
        self.last_trained_file = None

        if self.ner_enabled:
            self.ner_tokens = data_vocab_dicts["NER_tokens"]
            self.ner_label_to_token_dict = data_vocab_dicts["NER_label_to_token_dict"]

        self._encoded_x1, self._encoded_x2, self._encoded_y = None, None, None
        self._train_QA_pairs = []

    def __str__(self):
        return f"ChatBot Object: N_in={self.n_in}, N_out={self.n_out}, Vocab Size={self.vocab_size},\
                 Vocab File={self.vocab_file}, NER={self.ner_enabled}, Training Data={self.last_trained_file}."

    def __bool__(self):
        return self.encoder is not None and self.decoder is not None

    @staticmethod
    def _create_and_cache_vocab(vocab_file, vocab_size, ner_enabled=True):
        """
        Private Static Method.

        Creates and pickles a vocab from VOCAB_FILE. Note that VOCAB_FILE
        is expected to come as a json file where said file has a list of
        question-answer pairs saved as 'vocab_data':
            For example:
                [...,["Did you change your hair?", "No."], ["Hi!", "Hello."],...]
        Note that said json file needs to be in the same dir as this script.

        TODO: create a handel for vocab files that are NOT question answer pairs.
            MAKE THIS A JSON FILE EDIT CHANGE on the vocab_data selection......

        Vocab uses most frequent words first when truncating the vocab to
        fit the vocab size.

        Note that the cached vocab also saves a set of NER tokens (from the given
        vocab file) for future references.

        This function is very expensive due to the NER tagging.

        TO SELF: This should be improved in the future to incorperate a better NER tagger
        that takes advantage for the Cornell DB structure. I.E NLP the whole movie and
        tag from there... CAN BE DONE WITH DIFFERENT JASON FORMAT.

        :param vocab_file: file name of the jason file containing vocab data.
        :param vocab_size: the fixed size of the vocab.
        :param ner_enabled: toggles NER encoding for vocab.
        """
        word_freq, i = {}, 0
        vocab_data = json.load(open(vocab_file))["vocab_data"]
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
                if bool(re.search('[a-zA-Z]', tok)) and tok not in ner_tokens:
                    tok = tok.lower()
                    if tok in word_freq:
                        word_freq[tok] += 1
                    else:
                        word_freq[tok] = 1
            i += 1
            sys.stdout.write(f"\rCreating Vocab, parsed {i}/{len(vocab_data)} lines of vocab data.")
            sys.stdout.flush()

        special_tokens = ["<PADD>", "<START>", "<UNK>"] + list(ner_label_tokens)

        vocab = sorted(list(word_freq.keys()), key=lambda w: word_freq.get(w, 0), reverse=True)
        vocab = special_tokens + vocab[:vocab_size - len(special_tokens)]

        np.random.shuffle(vocab)  # Shuffle for validation check of cached files.

        if ner_enabled:
            dump = {"word_to_id": {c: i for i, c in enumerate(vocab)},
                    "id_to_word": {i: c for i, c in enumerate(vocab)},
                    "NER_tokens": ner_tokens, "NER_label_to_token_dict": ner_label_to_token_dict}
        else:
            dump = {"word_to_id": {c: i for i, c in enumerate(vocab)},
                    "id_to_word": {i: c for i, c in enumerate(vocab)}}
        pickle.dump(dump, open("cache/vocab.pickle", 'wb'))
        print(f"\nCached vocab file. Vocab size = {vocab_size}, Vocab Data = {vocab_file}")
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
        sentence_tokens = list(filter(lambda s: bool(re.search('[a-zA-Z]', s)), nltk.word_tokenize(sentence)))
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

    def has_valid_encodings(self):
        """
        Check if the current instance has encodings that uses the instance's vocab.
        """
        if self._encoded_x1 is None or self._encoded_x2 is None \
                or self._encoded_y is None or self._train_QA_pairs == []:
            return False
        if len(self._encoded_x1) != len(self._encoded_x2) != len(self._encoded_y):
            return False
        arr = np.random.choice(len(self._encoded_x1), size=min(10, len(self._encoded_x1)), replace=False)
        for i in arr:
            question_str, answer_str = self._train_QA_pairs[i]

            q_vec_ref = self.vectorize(question_str, self.n_in)
            a_vec_ref = self.vectorize(answer_str, self.n_out)
            a_shift_vec_ref = np.roll(a_vec_ref, 1)
            a_shift_vec_ref[0] = self.word_to_id_dict["<START>"]

            q_vec = self._encoded_x1[i]
            a_vec = self._encoded_y[i]
            a_shift_vec = self._encoded_x2[i]

            if q_vec != q_vec_ref or a_vec != a_vec_ref or a_shift_vec != a_shift_vec_ref:
                return False
        return True

    def _create_and_save_encoding(self, training_data_pairs, verbose=0):
        """
        Private method for the 'train' & 'batch_generator' methods of this class.

        This method creates and saves an 'encoded' version of the training data
        (from DATA_FILE) so that the generator does not have to do this work
        every time it is called.

        Said list of encodings are saved as private instance attrs.

        The 'encoding' is as follows:
            Given a sentence, we create an array/vector of size N_in or N_out
            (depending on which data we are encoding) where element i of
            said vector is token i's word ID in the word_to_id_dict dictionary
            of this object.

        Note each question in the training data gets its own vector & file and
        each answer gets 2 vectors & files (one of them is shifted by 1 time step).

        :param training_data_pairs: A list of question answer pairs as training data.
        :param verbose: update messages during execution.
        :return: The number question-answer pairs encoded.
        """
        def is_valid_data(question, answer):
            q_toks = [tok for tok in nltk.word_tokenize(question) if bool(re.search('[a-zA-Z]', tok))]
            a_toks = [tok for tok in nltk.word_tokenize(answer) if bool(re.search('[a-zA-Z]', tok))]
            return len(q_toks) <= self.n_in and len(a_toks) <= self.n_out

        encoded_x1, encoded_x2, encoded_y = [], [], []
        train_QA_pairs = []

        print("-==Encoding Training Data==-")
        for i, (q, a) in enumerate(training_data_pairs):
            if is_valid_data(q, a):
                train_QA_pairs.append((q, a))
                q_vec = self.vectorize(q, self.n_in)
                a_vec = self.vectorize(a, self.n_out)
                a_shift_vec = np.roll(a_vec, 1)
                a_shift_vec[0] = self.word_to_id_dict["<START>"]
                encoded_x1.append(q_vec)
                encoded_y.append(a_vec)
                encoded_x2.append(a_shift_vec)
            if verbose:
                sys.stdout.write(f"\rProcessed {i}/{len(training_data_pairs)} Question-Answer Pairs.")
                sys.stdout.flush()
        print("")

        self._encoded_x1, self._encoded_x2, self._encoded_y = encoded_x1, encoded_x2, encoded_y
        self._train_QA_pairs = train_QA_pairs

        pickle.dump(encoded_x1, open("cache/x1.pickle", 'wb'))
        pickle.dump(encoded_x2, open("cache/x2.pickle", 'wb'))
        pickle.dump(encoded_y, open("cache/y.pickle", 'wb'))
        pickle.dump(train_QA_pairs, open("cache/train_QA_pairs.pickle", 'wb'))

        print("Cached encoded training data.")
        return True

    def batch_generator(self, batch_size=32):
        """
        A generator that generates a list (length = BATCH_SIZE) of one-hot encoded
        vectors of the training data at each yield.

        Each batch is a randomized selection of un-yielded training data.

        IMPORTANT:
        This generator relies on the private method '_create_and_save_encoding' being
        called once before this generator's call as it requires the encodings that
        said method creates.

        :param batch_size: The size of the batch used in training.
        :return: The number question-answer pairs encoded.
        """
        if not self._encoded_x1 or not self._encoded_x2 or not self._encoded_y:
            raise RuntimeError("Attempted to generate one-hot encodings without encoded training data.")

        lst = list(range(len(self._encoded_y)))
        np.random.shuffle(lst)
        queue = deque(lst)
        batch_num = 0

        while queue:
            this_batch_size = min(batch_size, len(queue))
            X_1_encoded = np.empty(this_batch_size, dtype=bytearray)
            X_2_encoded = np.empty(this_batch_size, dtype=bytearray)
            Y_encoded = np.empty(this_batch_size, dtype=bytearray)
            for i in range(this_batch_size):
                encoded_index = queue.pop()
                X_1_encoded[i] = self._encoded_x1[encoded_index]
                X_2_encoded[i] = self._encoded_x2[encoded_index]
                Y_encoded[i] = self._encoded_y[encoded_index]
            X_1 = self._array_one_hot_encode(X_1_encoded, self.n_in, len(self.word_to_id_dict))
            X_2 = self._array_one_hot_encode(X_2_encoded, self.n_out, len(self.word_to_id_dict))
            Y = self._array_one_hot_encode(Y_encoded, self.n_out, len(self.word_to_id_dict))
            batch_num += 1
            yield X_1, X_2, Y, f"{batch_num}/{int(np.ceil(len(lst)/batch_size))}"

    def train(self, data_file, epoch, batch_size=32, split_percentage=0.35, force_encode=False, verbose=0):
        """
        Trains the chatbot's encoder and decoder LSTMs (= the Seq2Seq model).

        Note that DATA_FILE is expected to come as a json file where said
        file is has a list of question-answer pairs saved as 'question_answer_pairs'.
            Said list has the following form:
                [...,["Did you change your hair?", "No."], ["Hi!", "Hello."],...]
        Note that said file needs to be in the same dir as this script.

        :param data_file: The json file containing the question-answer pairs.
        :param epoch: number of epochs in training.
        :param batch_size: size of the batch in training.
        :param split_percentage: a float between 0 and 1. It is the percentage of
                                 training data held out for validation.
        :param force_encode: Forces the script to re-encode the training data, even
                             if there is a cached copy.
        :param verbose: update messages during training.
        """
        self.last_trained_file = data_file

        model, encoder, decoder = define_models(len(self.word_to_id_dict), len(self.id_to_word_dict), 128)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        if verbose:
            print(model.summary())

        if not force_encode and not self.ignore_cache and os.path.isfile("cache/x1.pickle") \
                and os.path.isfile("cache/x2.pickle") and os.path.isfile("cache/y.pickle")\
                and os.path.isfile("cache/train_QA_pairs.pickle"):
            self._encoded_x1 = pickle.load(open("cache/x1.pickle", 'rb'))
            self._encoded_x2 = pickle.load(open("cache/x2.pickle", 'rb'))
            self._encoded_y = pickle.load(open("cache/y.pickle", 'rb'))
            self._train_QA_pairs = pickle.load(open("cache/train_QA_pairs.pickle", 'rb'))

        if self.has_valid_encodings():
            print("[!] Using cached training data encodings.")
        else:
            self._create_and_save_encoding(training_data_pairs=json.load(open(data_file))["question_answer_pairs"],
                                           verbose=verbose)

        print(f"Size of encoded training data: {len(self._encoded_x1)}")
        print(f"-==Training==-\n\tEpochs: {epoch}, Batch Size: {batch_size}, "
              f"Question-Answer Pairs: {len(self._encoded_y)}.\n")

        for ep in range(epoch):
            batch_gen = self.batch_generator(batch_size=batch_size)
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
        return True

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
    # TODO: train a model where N_in = N_out.
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
    opts.add_option("-I", '--ignore_cache', action="store_true", dest="ignore_cached",
                    help="Forces the script to ignore the cached files.")
    opts.add_option("-E", '--encode_training_data', action="store_true", dest="encode_training_data",
                    help="Forces the script to ignore the cached encoding files,")
    opts.add_option("-N", '--NER_enabled', action="store_true", dest="NER_enabled",
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
    # TODO: HIGHARCHICAL STRUCTURE FOR PUBLISHING...
    # TODO: refactor for efficientcy and clenlyness and names of files...
    # TODO: Documentation & REFACTOR TO REMOVE REDUNDANT INFO.
    # TODO: First Time installer...
    # TODO: publish README...
    opts = get_options()
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
                           opts.ignore_cached, opts.NER_enabled)
        chat_bot.train(opts.data_file, opts.epoch, opts.batch_size, opts.split,
                       opts.encode_training_data, opts.verbose)

        if new_model_name:
            new_model_path = f"{opts.saved_models_dir}/{new_model_name}"
            if not os.path.exists(new_model_path):
                os.mkdir(new_model_path)
            pickle.dump(chat_bot, open(f"{new_model_path}/chatbot.pickle", 'wb'))
            if not os.path.exists(f"{new_model_path}/backup"):
                os.mkdir(f"{new_model_path}/backup")
            chat_bot.encoder.save_weights(f"{new_model_path}/backup/encoder.h5")
            chat_bot.decoder.save_weights(f"{new_model_path}/backup/decoder.h5")
            shutil.copytree("cache", f"{new_model_path}/backup")
            shutil.copyfile(opts.data_file, f"{new_model_path}/backup/[TRAIN_DATA]{opts.data_file}")
            shutil.copyfile(opts.vocab_file, f"{new_model_path}/backup/[VOCAB_DATA]{opts.vocab_file}")
            print(f"\nSaved the trained model to: '{new_model_path}'.")
    chat_bot.chat()
