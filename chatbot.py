import os
import sys
import pickle
import itertools
import spacy
import shutil
import json
import nltk
import numpy as np
from collections import deque
from optparse import OptionParser
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense

NER_NLP = spacy.load('en')


class ChatBot:

    def __init__(self, n_in, n_out, vocab_size, vocab_file, ignore_cache, ner_enabled):
        if not os.path.exists("cache"):
            os.makedirs("cache")

        # Common instance attrs.
        self.n_in, self.n_out = n_in, n_out
        self.ignore_cache = ignore_cache
        self.ner_enabled = ner_enabled
        self.vocab_file = vocab_file
        self.vocab_file_sig = f"{vocab_file} (last_mod: {os.path.getmtime(vocab_file)})"
        self.train_data_file = None
        self.train_data_file_sig = None
        self.train_data_filter_mode = 0
        self.model, self.encoder, self.decoder = None, None, None

        # Vocab creation.
        if os.path.isfile("cache/vocab.pickle") and not ignore_cache:
            try:
                self.vocab_data_dict = pickle.load(open("cache/vocab.pickle", 'rb'))
                if len(self.vocab_data_dict["token_to_id"]) != len(self.vocab_data_dict["token_to_id"]):
                    raise ValueError("Cached vocab's dictionary lengths do not match.")
                if self.vocab_data_dict["vocab_size"] != vocab_size \
                        and len(self.vocab_data_dict["token_to_id"]) != vocab_size:
                    raise ValueError(f"Cached vocab size does not match.")
                if self.vocab_file_sig != self.vocab_data_dict["signature"]:
                    raise ValueError(f"{self.vocab_file} is not the source file of 'cache/vocab.pickle'.")
                if self.ner_enabled and ("NER_tokens" not in self.vocab_data_dict.keys()
                                         or "NER_label_to_token_dict" not in self.vocab_data_dict.keys()):
                    raise ValueError("Cached vocab does not contain NER data.")
                print("[!] Using cached vocab.")
            except Exception as e:
                print(f"Exception encountered when reading vocab data: {type(e).__name__}, {e}")
                self.vocab_data_dict = self._create_and_cache_vocab(vocab_size)
        else:
            print("No cached vocab file found.")
            self.vocab_data_dict = self._create_and_cache_vocab(vocab_size)
        self.token_to_id_dict = self.vocab_data_dict["token_to_id"]
        self.id_to_token_dict = self.vocab_data_dict["id_to_token"]
        self.vocab_size = len(self.token_to_id_dict) if vocab_size is None else vocab_size

        # NER instance attrs that depend on the vocab.
        if self.ner_enabled:
            self.ner_tokens = self.vocab_data_dict["NER_tokens"]
            self.ner_label_to_token_dict = self.vocab_data_dict["NER_label_to_token_dict"]

        # Training attrs.
        self._v_encoded_x1, self._v_encoded_x2, self._v_encoded_y = None, None, None
        self._trained_QA_pairs = []

    def __repr__(self):
        return f"<ChatBot Object: Vocab Size={self.vocab_size}, N_in={self.n_in}, N_out={self.n_out}," \
            f" Vocab File={self.vocab_file_sig}, NER={self.ner_enabled}," \
            f" Training Data={self.train_data_file_sig}, Filter Mode={self.train_data_filter_mode}>"

    def __bool__(self):
        return self.encoder is not None and self.decoder is not None

    @staticmethod
    def _create_validation_split(X_1, X_2, Y, percentage):
        """
        Private method use in the train method.

        Creates a validation split of X_1, X_2 and Y.

        This requires X_1, X_2 and Y to have at least 2 elements.
        And assumes X_1, X_2 and Y have the same number of elements.
        """
        v_count = int(np.math.floor(len(X_1)*percentage))
        v_indices = set(np.random.choice(len(X_1), size=v_count, replace=False))
        X_1t, X_2t, Y_t = [], [], []
        X_1v, X_2v, Y_v = [], [], []

        for i in range(len(X_1)):
            if i in v_indices:
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

    @staticmethod
    def _create_sampling_validation_split(X_1, X_2, Y, percentage):
        """
        Private method use in the train method.

        Creates a validation split of X_1, X_2 and Y based on a sampling
        scheme instead of a deterministic scheme. So, each element of X_1
        X_2 and Y have a PERCENTAGE chance of being use as validation data.

        This requires X_1, X_2 and Y to have at least 2 elements.
        And assumes X_1, X_2 and Y have the same number of elements.
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

    def _create_and_cache_vocab(self, vocab_size):
        """
        Private Method used in the constructor.

        Creates and caches a vocab from this instance's vocab file. Note that the
        vocab file is expected to come as a .json file where the vocab data is
        saved on row: 'vocab_data'. The vocab data is either a list of question-answer
        pairs/tuples or a list of sentences/strings.

        The created vocab uses the most frequent tokens first when truncating the
        vocab to fit the vocab size.

        Note that this method also saves a set of NER tokens (from the given
        vocab file) for future references if NER is enabled.

        This method can take a while if NER is enabled.
        # TODO: make this more efficient without using the nested functions... (? maybe)
        """
        count = 0
        tok_freq = {}
        ner_tokens = set()
        ner_label_tokens = set()
        ner_label_to_token_dict = {}
        vocab_data = json.load(open(self.vocab_file))["vocab_data"]

        def process_for_entities(nlp_entities):
            for entity in nlp_entities:
                ner_label_tokens.add(f"<{entity.label_}>")
                for tok in nltk.word_tokenize(entity.text):
                    ner_tokens.add(tok)
                    label = f"<{entity.label_}>"
                    if label in ner_label_to_token_dict:
                        ner_label_to_token_dict[label].add(tok)
                    else:
                        ner_label_to_token_dict[label] = {tok}

        def process_for_freq(tokens):
            for tok in tokens:
                if tok not in ner_tokens:
                    tok = tok.lower()
                    if tok in tok_freq:
                        tok_freq[tok] += 1
                    else:
                        tok_freq[tok] = 1

        def progress_message():
            nonlocal count
            count += 1
            sys.stdout.write(f"\rCreating Vocab, parsed {count}/{len(vocab_data)} lines of vocab data.")
            sys.stdout.flush()

        try:
            first_vocab_el = vocab_data[0]
        except IndexError:
            raise ValueError(f"{self.vocab_file} contains no data.")

        if type(first_vocab_el) == str:
            for line in vocab_data:
                if self.ner_enabled:
                    process_for_entities(NER_NLP(line).ents)
                process_for_freq(nltk.word_tokenize(line))
                progress_message()
        elif (type(first_vocab_el) == list or type(first_vocab_el) == tuple) and len(first_vocab_el) == 2:
            for question, answer in vocab_data:
                if self.ner_enabled:
                    process_for_entities(itertools.chain(NER_NLP(question).ents, NER_NLP(answer).ents))
                process_for_freq(itertools.chain(nltk.word_tokenize(question), nltk.word_tokenize(answer)))
                progress_message()
        else:
            raise ValueError(f"Vocab data: '{vocab_data}' is not supported.")

        # Hardcoded special tokens. DO NOT change the order of PADD, START and UNK.
        special_tokens = ["<PADD>", "<START>", "<UNK>"] + list(ner_label_tokens)
        top_vocab_toks = sorted(list(tok_freq.keys()), key=lambda w: tok_freq.get(w, 0), reverse=True)
        if vocab_size is not None:
            top_vocab_toks = top_vocab_toks[:vocab_size - len(special_tokens)]
        np.random.shuffle(top_vocab_toks)  # Shuffle for validation check of cached files.
        vocab = special_tokens + top_vocab_toks

        if self.ner_enabled:
            dump = {"signature": self.vocab_file_sig,
                    "vocab_size": vocab_size,
                    "token_to_id": {c: i for i, c in enumerate(vocab)},
                    "id_to_token": {i: c for i, c in enumerate(vocab)},
                    "NER_tokens": ner_tokens, "NER_label_to_token_dict": ner_label_to_token_dict}
        else:
            dump = {"signature": self.vocab_file_sig,
                    "vocab_size": vocab_size,
                    "token_to_id": {c: i for i, c in enumerate(vocab)},
                    "id_to_token": {i: c for i, c in enumerate(vocab)}}
        pickle.dump(dump, open("cache/vocab.pickle", 'wb'))
        print(f"\nCached vocab file. Vocab size = {vocab_size}, Vocab Data = {self.vocab_file}")
        self.vocab_data_dict = dump
        return dump

    def define_models(self, latent_dim, verbose):
        """
        Define the encoder and decoder models for this object.

        :param latent_dim: The dimensionality of the Encoder and Decoder's LSTM.
        :param verbose: Print model summary if this is truthy.
        """
        assert len(self.token_to_id_dict) == len(self.id_to_token_dict)

        encoder_inputs = Input(shape=(None, len(self.token_to_id_dict)))
        encoder = LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None, len(self.token_to_id_dict)))
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(len(self.token_to_id_dict), activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        self.encoder = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

        self.model.compile(optimizer='adam', loss='categorical_crossentropy')

        if verbose:
            print(self.model.summary())

        return True

    def v_encode(self, sentence, length=None):
        """
        Vocab encode the SENTENCE into a vector with LENGTH elements.

        Note that this is NOT a one-hot encoding. Instead, it uses the vocab
        to create an encoded vector where entry i of said encoding is the
        token id of L[i] (L = list of tokens of SENTENCE).

        :param sentence: A string that is to be encoded. Note that it CAN include
                         punctuation and unknown words/tokens.
        :param length: The length of the returned vector. It defaults to the number
                       of tokens in SENTENCE.
        :return: The vocab vector encoding of the sentence as described above.
        """
        sentence = sentence.strip()
        sentence_tokens = nltk.word_tokenize(sentence)
        length = length if length else len(sentence_tokens)
        vector = np.zeros(length, dtype=int)  # 0 = token id for '<PADD>'.

        entity = {}
        if self.ner_enabled and any(w for w in sentence_tokens if w in self.ner_tokens):
            for ent in NER_NLP(sentence).ents:
                for w in nltk.word_tokenize(ent.text):
                    entity[w] = f"<{ent.label_}>"

        for i, tok in zip(range(length), sentence_tokens):
            if tok in entity:
                tok_id = self.token_to_id_dict[entity[tok]]
            else:
                tok_id = self.token_to_id_dict.get(tok.lower(), 2)  # 2 = token id for '<UNK>'.
            vector[i] = tok_id
        return vector

    def _one_hot_encode(self, vocab_encoded_vector):
        """
        Private method for train and decoder/predictor methods.

        This creates the true one hot encoded tensor from the VOCAB_ENCODED_VECTOR.

        A vocab encoded vector is a vector where element i is the token index
        (from the token_to_id dict of this obj) of token i of the sentence
        that said vector is encoding.

        :param vocab_encoded_vector: The encoded vector described above.
        """
        if len(vocab_encoded_vector) > 0:
            n = len(vocab_encoded_vector[0])
            encoding = np.zeros((len(vocab_encoded_vector), n, len(self.token_to_id_dict)))
            for i, seq in enumerate(vocab_encoded_vector):
                if len(seq) != n:
                    raise RuntimeError("Vocab encoded sentences do not have matching dimensions.")
                for j, tok_id in enumerate(seq):
                    encoding[i, j, tok_id] = 1
            return encoding

    def _create_and_cache_vocab_encoding(self, training_data_pairs, verbose=0):
        """
        Private method used in the 'train' & 'batch_generator' methods of this class.

        This method creates and caches 3 matrices of 'vocab encoded' training data
        (training data is from TRAINING_DATA_PAIRS) for the training data generator.

        :param training_data_pairs: A list of question answer pairs as training data.
        :param verbose: update messages during execution.
        """
        def is_valid_data(question, answer):
            q_toks, a_toks = nltk.word_tokenize(question), nltk.word_tokenize(answer)
            if self.train_data_filter_mode == 1 and '?' not in q_toks:
                return False
            if self.train_data_filter_mode == 2 and not('?' in q_toks and '?' in a_toks):
                return False
            return len(q_toks) <= self.n_in and len(a_toks) <= self.n_out

        encoded_x1, encoded_x2, encoded_y = [], [], []
        trained_QA_pairs = []

        print("-==Vocab Encoding Training Data==-")
        for i, (q, a) in enumerate(training_data_pairs):
            if is_valid_data(q, a):
                q_vec = self.v_encode(q, self.n_in)
                a_vec = self.v_encode(a, self.n_out)
                a_shift_vec = np.roll(a_vec, 1)
                a_shift_vec[0] = 1  # 1 = token id for '<START>'

                trained_QA_pairs.append((q, a))
                encoded_x1.append(q_vec)
                encoded_y.append(a_vec)
                encoded_x2.append(a_shift_vec)
            if verbose:
                sys.stdout.write(f"\rProcessed {i+1}/{len(training_data_pairs)} Question-Answer Pairs.")
                sys.stdout.flush()
        print("")

        self._v_encoded_x1, self._v_encoded_x2, self._v_encoded_y = encoded_x1, encoded_x2, encoded_y
        self._trained_QA_pairs = trained_QA_pairs

        pickle.dump({"signature": self.train_data_file_sig, "filter": self.train_data_filter_mode,
                     "data": encoded_x1}, open("cache/x1.pickle", 'wb'))
        pickle.dump({"signature": self.train_data_file_sig, "filter": self.train_data_filter_mode,
                     "data": encoded_x2}, open("cache/x2.pickle", 'wb'))
        pickle.dump({"signature": self.train_data_file_sig, "filter": self.train_data_filter_mode,
                     "data": encoded_y}, open("cache/y.pickle", 'wb'))
        pickle.dump({"signature": self.train_data_file_sig, "filter": self.train_data_filter_mode,
                     "data": trained_QA_pairs}, open("cache/trained_QA_pairs.pickle", 'wb'))

        print("Cached vocab encoded training data.")
        return True

    def batch_generator(self, batch_size=32):
        """
        A generator that generates a list (length = BATCH_SIZE) of one-hot encoded
        vectors of the training data at each yield.

        Each batch is created from a randomized selection of un-yielded training data.

        IMPORTANT:
        This generator relies on the private method: '_create_and_cache_vocab_encoding'
        being called once before this generator's call as it requires the 3 vocab encoded
        training data matrices that said method generates.

        :param batch_size: The size of each batch yielded.
        """
        if not self._v_encoded_x1 or not self._v_encoded_x2 or not self._v_encoded_y:
            raise RuntimeError("Attempted to generate one-hot encodings without vocab encoded training data.")

        lst = list(range(len(self._v_encoded_y)))
        np.random.shuffle(lst)
        queue = deque(lst)
        batch_num = 0
        total_batch_count = int(np.ceil(len(lst)/batch_size))

        while queue:
            this_batch_size = min(batch_size, len(queue))
            X_1_encoded = np.empty(this_batch_size, dtype=bytearray)
            X_2_encoded = np.empty(this_batch_size, dtype=bytearray)
            Y_encoded = np.empty(this_batch_size, dtype=bytearray)

            if this_batch_size == 1:
                index = queue.pop()
                queue.extend([index, index])
                this_batch_size = 2

            for i in range(this_batch_size):
                encoded_index = queue.pop()
                X_1_encoded[i] = self._v_encoded_x1[encoded_index]
                X_2_encoded[i] = self._v_encoded_x2[encoded_index]
                Y_encoded[i] = self._v_encoded_y[encoded_index]

            X_1 = self._one_hot_encode(X_1_encoded)
            X_2 = self._one_hot_encode(X_2_encoded)
            Y = self._one_hot_encode(Y_encoded)

            batch_num += 1
            yield X_1, X_2, Y, f"{batch_num}/{total_batch_count}"

    def has_valid_vocab_encodings(self):
        """
        Check if the current instance has vocab encodings that uses the instance's vocab.

        An encoding is valid if this object's vocab was used for this object's encodings.

        The validation uses an idea similar to the bloom filter.
        The 'hash' portion of this bloom filter idea is solely dependent on numpy's
        shuffle of the vocab.

        Note that there is a slim chance of a false positive. This chance can be lowered
        by increasing N.
        """
        if self._v_encoded_x1 is None or self._v_encoded_x2 is None \
                or self._v_encoded_y is None or not self._trained_QA_pairs:
            return False
        if len(self._v_encoded_x1) != len(self._v_encoded_x2) != len(self._v_encoded_y):
            return False

        N = 25
        arr = np.random.choice(len(self._v_encoded_x1), size=min(N, len(self._v_encoded_x1)), replace=False)
        for i in arr:
            question_str, answer_str = self._trained_QA_pairs[i]

            q_vec_ref = self.v_encode(question_str, self.n_in)
            a_vec_ref = self.v_encode(answer_str, self.n_out)
            a_shift_vec_ref = np.roll(a_vec_ref, 1)
            a_shift_vec_ref[0] = 1  # 1 = token id for '<START>'.

            q_vec = self._v_encoded_x1[i]
            a_vec = self._v_encoded_y[i]
            a_shift_vec = self._v_encoded_x2[i]

            if not np.array_equal(q_vec, q_vec_ref) or not np.array_equal(a_vec, a_vec_ref) \
                    or not np.array_equal(a_shift_vec, a_shift_vec_ref):
                return False
        return True

    def _load_cached_v_encoded_train_data(self):
        """
        Private method to load the encoded training data from cache.
        Does not load if the signature or filter does not match this object's
        data file or filter setting (respectively).
        """
        if not self.ignore_cache and os.path.isfile("cache/x1.pickle") \
                and os.path.isfile("cache/x2.pickle") and os.path.isfile("cache/y.pickle") \
                and os.path.isfile("cache/trained_QA_pairs.pickle"):
            x1_file_dict = pickle.load(open("cache/x1.pickle", 'rb'))
            x2_file_dict = pickle.load(open("cache/x2.pickle", 'rb'))
            y_file_dict = pickle.load(open("cache/y.pickle", 'rb'))
            QA_file_dict = pickle.load(open("cache/trained_QA_pairs.pickle", 'rb'))

            try:
                if not (self.train_data_file_sig == x1_file_dict["signature"] == x2_file_dict["signature"]
                        == x2_file_dict["signature"] == y_file_dict["signature"] == QA_file_dict["signature"]):
                    raise ValueError("Miss matched source of cached file.")
                if not (self.train_data_filter_mode == x1_file_dict['filter'] == x2_file_dict['filter']
                        == x2_file_dict['filter'] == y_file_dict['filter'] == QA_file_dict['filter']):
                    raise ValueError("Miss matched data filter of cached file.")
                self._v_encoded_x1 = x1_file_dict['data']
                self._v_encoded_x2 = x2_file_dict['data']
                self._v_encoded_y = y_file_dict['data']
                self._trained_QA_pairs = QA_file_dict['data']
                return True
            except (ValueError, AttributeError, TypeError, KeyError) as e:
                print(f"Exception when loading cached training data: {type(e).__name__}, {e}")
        return False

    def train(self, data_file, filter_mode, latent_dim, epoch, batch_size, split_pct, verbose):
        """
        Trains the chatbot's encoder and decoder LSTMs (= the Seq2Seq model).

        Note that DATA_FILE is expected to come as a json file where said
        file is has a list of question-answer pairs saved on row: 'question_answer_pairs'.
            Said list has the following form:
                [...,["Did you change your hair?", "No."], ["Hi!", "Hello."],...]

        Filter Modes:
            0) Only take Questions that have N_in number of tokens and only take
            Answers that have N_out number of tokens.
            1) Mode 0 AND Question must have a '?' token.
            2) Mode 0 AND Question & Answer must have a '?' token.

        :param data_file: The json file containing the question-answer pairs.
        :param filter_mode: An int that determines the filter mode of the training data.
        :param latent_dim: The dimensionality of the Encoder and Decoder's (the model's) LSTM.
        :param epoch: number of epochs in training.
        :param batch_size: size of the batch in training.
        :param split_pct: a float between 0 and 1. It is the percentage of training data
                          that is held out for validation.
        :param verbose: Boolean to enable/disable update messages during training.
        """
        assert len(self.token_to_id_dict) == len(self.id_to_token_dict)

        self.train_data_file_sig = f"{data_file} (last_mod: {os.path.getmtime(data_file)})"
        self.train_data_file = data_file
        self.train_data_filter_mode = filter_mode

        if not self.ignore_cache:
            self._load_cached_v_encoded_train_data()

        if not self.ignore_cache and self.has_valid_vocab_encodings():
            print("[!] Using cached training data encodings.")
        else:
            data_pairs = json.load(open(self.train_data_file))["question_answer_pairs"]
            self._create_and_cache_vocab_encoding(training_data_pairs=data_pairs, verbose=verbose)

        print(f"-==Training on {len(self._v_encoded_y)} Question-Answer pairs==-")

        self.define_models(latent_dim, verbose=verbose)

        for ep in range(epoch):
            batch_gen = self.batch_generator(batch_size=batch_size)
            for X_1, X_2, Y, batch_counter in batch_gen:
                if verbose:
                    sys.stdout.flush()
                    sys.stdout.write('\x1b[2K')
                    print(f"\rEpoch: {ep}/{epoch}, Batch: {batch_counter}. \tTraining...")

                X_1t, X_2t, Y_t, X_1v, X_2v, Y_v = ChatBot._create_validation_split(X_1, X_2, Y, split_pct)
                self.model.fit([X_1t, X_2t], Y_t, epochs=1, batch_size=batch_size,
                               validation_data=([X_1v, X_2v], Y_v), verbose=verbose)

        if verbose:
            print(f"Training Complete.\nTrained on {len(self._v_encoded_y)} Question-Answer pairs")
        return True

    def _predict(self, X_in):
        """ Private method used for main chat loop.
        """
        curr_in_state = self.encoder.predict(X_in)
        curr_out_state = [
            self._one_hot_encode([[1]])  # 1 = token id for '<START>'.
        ]
        Y_hat = []
        for t in range(self.n_in):
            prediction, h, c = self.decoder.predict(curr_out_state + curr_in_state)
            curr_in_state = [h, c]
            curr_out_state = [prediction]
            Y_hat.append(prediction)
        return np.array(Y_hat)

    def _vector_to_sentece(self, vector):
        """ Private method used for main chat loop.
        """
        sentence_tokens = []
        for el in vector:
            tok_id = np.argmax(el)  # Fetch index that has 1 as element.
            tok = self.id_to_token_dict.get(tok_id, "<UNK>")
            if tok == "<PADD>":
                return " ".join(sentence_tokens)
            sentence_tokens.append(tok)
        return " ".join(sentence_tokens)

    def chat(self):
        """ Main chat loop with the chatbot.
        """
        if not self:
            raise RuntimeError("Attempted to chat with an untrained model.")
        print("Chat Bot ready, type anything to start: (Ctrl + C to exit)")
        try:
            while True:
                sys.stdout.write(">")
                sys.stdout.flush()
                input_str = input()
                vocab_encoded_X_in = [self.v_encode(input_str, self.n_in)]
                X_in = self._one_hot_encode(vocab_encoded_X_in)
                Y_hat = self._predict(X_in)
                print("Response: {}".format(self._vector_to_sentece(Y_hat)))
                print(" ")
        except KeyboardInterrupt:
            print("Done Chatting...")

    def save(self, directory, verbose):
        """
        :param directory: The directory of where this object is going to be saved.
        :param verbose: update messages during execution.
        """
        if not os.path.exists(directory):
            os.mkdir(directory)
        else:
            shutil.rmtree(directory)
            os.mkdir(directory)
        pickle.dump(self, open(f"{directory}/chatbot.pickle", 'wb'))
        if not os.path.exists(f"{directory}/backup"):
            os.mkdir(f"{directory}/backup")
        self.encoder.save_weights(f"{directory}/backup/encoder.h5")
        self.decoder.save_weights(f"{directory}/backup/decoder.h5")
        shutil.copytree("cache", f"{directory}/backup/cache")
        shutil.copyfile(self.train_data_file, f"{directory}/backup/[T-DAT]{self.train_data_file}")
        shutil.copyfile(self.vocab_file, f"{directory}/backup/[V-DAT]{self.vocab_file}")
        if verbose:
            print(f"\nSaved the trained model to: '{directory}'.")
        return True


def get_options():
    opts = OptionParser()
    opts.add_option('-i', '--N_in', dest='N_in', type=int, default=10,
                    help="The number of time steps for the encoder. Default = 10.")
    opts.add_option('-o', '--N_out', dest='N_out', type=int, default=20,
                    help="The number of time setps for the decoder. Default = 20.")
    opts.add_option('-v', '--vocab_size', dest='vocab_size', type=int, default=None,
                    help='The size of the vocab of the Chatbot. Default = None')
    opts.add_option('-f', '--vocab_file', dest='vocab_file', type=str, default="Cornell_Movie_Dialogs_Data.json",
                    help="The directory of the file that is used to define the vocab. "
                         "This file must be a json file that contains a list of question-answer"
                         "pairs/lists. Reference the included/default file for details."
                         "Default = 'Cornell_Movie_Dialogs_Data.json'")
    opts.add_option("-I", '--ignore_cache', action="store_true", dest="ignore_cached",
                    help="Forces the script to ignore the cached files.")
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
    opts.add_option('-c', '--filter_mode', dest='filter_mode', type=int, default=0,
                    help="An integer that dictates the filter imposed of the data. MODES: {0, 1, 2}. "
                         "Mode 0: Only take Questions that have N_in number of tokens and only take Answers "
                         "that have N_out number of tokens. Mode 1: All of Mode 0 AND Questions must have a '?' token. "
                         "Mode 2: All of Mode 0 AND Question & Answer must have a '?' token. "
                         "Default = 0")
    opts.add_option('-e', '--epoch', dest='epoch', type=int, default=500,
                    help="The number of epochs for training. Default = 100.")
    opts.add_option('-b', '--batch_size', dest='batch_size', type=int, default=32,
                    help="The batch size for training. Default = 32.")
    opts.add_option('-l', '--latent_dim', dest='latent_dim', type=int, default=128,
                    help="The dimensionality of the Encoder and Decoder's LSTM. Default = 128.")
    opts.add_option('-s', '--split', dest='split', type=float, default=0.35,
                    help="The percentage (val between 0 - 1) of data held out for validation. "
                         "Default = 0.35")
    opts.add_option('-m', '--saved_models_dir', dest='saved_models_dir', type=str, default="saved_models",
                    help="The directory for all of the saved (trained) models. "
                         "Default = 'saved_models'")
    return opts.parse_args()[0]


def get_saved_model(directory):
    saved_models = os.listdir(directory)
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
    # TODO: First Time installer...
    # TODO: publish README...
    opts = get_options()
    if not os.path.exists(opts.saved_models_dir):
        os.makedirs(opts.saved_models_dir)

    sys.stdout.write("\rLoad a saved model? (y/n) ")
    sys.stdout.flush()
    user_input = input()

    saved_model_dir = get_saved_model(opts.saved_models_dir) if user_input[0].lower() == 'y' else None

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
        chat_bot.train(opts.data_file, opts.filter_mode, opts.latent_dim, opts.epoch,
                       opts.batch_size, opts.split, opts.verbose)

        if new_model_name:
            new_model_directory = f"{opts.saved_models_dir}/{new_model_name}"
            chat_bot.save(new_model_directory, opts.verbose)
    chat_bot.chat()
