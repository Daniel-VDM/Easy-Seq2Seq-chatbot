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
from argparse import ArgumentParser
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense

NER_NLP = spacy.load('en')


class ChatBot:

    def __init__(self, n_in, n_out, vocab_size, vocab_file, ignore_cache, ner_enabled, verbose, cache_dir=None):
        self._cache_dir = cache_dir if cache_dir is not None else 'cache'
        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)

        # Common instance attrs.
        self.n_in, self.n_out = n_in, n_out
        self.ignore_cache = ignore_cache
        self.ner_enabled = ner_enabled
        self.vocab_file = vocab_file
        self.vocab_file_sig = f"{vocab_file} (last_mod: {os.path.getmtime(vocab_file)})"
        self.train_data_file = None
        self.train_data_file_sig = None
        self.train_data_filter_mode = 0
        self.train_cache_id = self.cache_id()
        self.model, self.encoder, self.decoder = None, None, None

        # Vocab creation.
        if os.path.isfile("cache/vocab.pickle") and not ignore_cache:
            try:
                self.vocab_data_dict = pickle.load(open(f"{self._cache_dir}/vocab.pickle", 'rb'))
                if len(self.vocab_data_dict["token_to_id"]) != len(self.vocab_data_dict["token_to_id"]):
                    raise ValueError("Cached vocab's dictionary lengths do not match.")
                if self.vocab_data_dict["vocab_size"] != vocab_size \
                        and len(self.vocab_data_dict["token_to_id"]) != vocab_size:
                    raise ValueError(f"Cached vocab size does not match.")
                if self.vocab_file_sig != self.vocab_data_dict["signature"]:
                    raise ValueError(f"Cached vocab signature does not match given vocab file.")
                if self.ner_enabled and ("NER_tokens" not in self.vocab_data_dict.keys()
                                         or "NER_label_to_token_dict" not in self.vocab_data_dict.keys()):
                    raise ValueError("Cached vocab does not contain NER data.")
                if verbose:
                    print("[!] Using cached vocab.")
            except Exception as e:
                if verbose:
                    print(f"Exception encountered when reading vocab data: {type(e).__name__}, {e}")
                self.vocab_data_dict = self._create_and_cache_vocab(vocab_size, verbose=verbose)
        else:
            if verbose:
                print("No cached vocab file found.")
            self.vocab_data_dict = self._create_and_cache_vocab(vocab_size, verbose=verbose)
        self.token_to_id_dict = self.vocab_data_dict["token_to_id"]
        self.id_to_token_dict = self.vocab_data_dict["id_to_token"]
        self.vocab_size = len(self.token_to_id_dict)  # Ignore param and use vocab dict for efficiency.

        # NER instance attrs that depend on the vocab.
        if self.ner_enabled:
            self.ner_tokens = self.vocab_data_dict["NER_tokens"]
            self.ner_label_to_token_dict = self.vocab_data_dict["NER_label_to_token_dict"]

        # Training attrs.
        self._v_encoded_x1, self._v_encoded_x2, self._v_encoded_y = [], [], []
        self._trained_QA_pairs = []

    def __repr__(self):
        return f"<ChatBot Instance: Vocab Size={self.vocab_size}, n_in={self.n_in}, n_out={self.n_out}," \
               f" Vocab File={self.vocab_file_sig}, NER={self.ner_enabled}," \
               f" Training File={self.train_data_file_sig}, Filter Mode={self.train_data_filter_mode}>"

    def __bool__(self):
        return self.encoder is not None and self.decoder is not None

    @staticmethod
    def load(model_dir, options):
        """
        :param model_dir: The directory of the saved model. Must contain the
                          cache used for the instance that is loaded.
        :param options: The options namespace used for the instance that is loaded.
        :return: A chatbot object with the loaded model or none if an error occurred.
        """
        new_instance = ChatBot(options.n_in, options.n_out, options.vocab_size, options.vocab_file,
                               False, not options.NER_disable, False, f'{model_dir}/cache')
        new_instance.train(options.train_file, options.filter_mode, options.latent_dim, 0,
                           options.batch_size, options.split, False)
        if new_instance.has_valid_vocab_encodings():
            new_instance.encoder.load_weights(f'{model_dir}/encoder.h5')
            new_instance.decoder.load_weights(f'{model_dir}/decoder.h5')
            new_instance.ignore_cache = options.ignore_cached
            new_instance._cache_dir = 'cache'
            return new_instance
        else:
            print(f"model: {model_dir} cannot be loaded.")

    @staticmethod
    def _create_validation_split(X_1, X_2, Y, percentage):
        """
        Private method use in the train method.

        Creates a validation split of X_1, X_2 and Y.

        Note: X_1, X_2 and Y are a list of one-hot encoded sentences.

        This requires X_1, X_2 and Y to have at least 2 elements.
        And assumes X_1, X_2 and Y have the same number of elements.
        """
        v_count = int(np.math.floor(len(X_1) * percentage))
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

        Note: X_1, X_2 and Y are a list of one-hot encoded sentences.

        This requires X_1, X_2 and Y to have at least 2 elements.
        And assumes X_1, X_2 and Y have the same number of elements.
        """
        X_1t, X_2t, Y_t = [X_1[0]], [X_2[0]], [Y[0]]
        X_1v, X_2v, Y_v = [X_1[1]], [X_2[1]], [Y[1]]
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

    @staticmethod
    def create_vocab_asset(vocab_file, NER_on, verbose):
        """
        Creates vocab asset from VOCAB_FILE. Note that the VOCAB_FILE is
        expected to come as a JSON file where the vocab data is saved as
        attribute: 'vocab_data'. The vocab data is either a list of
        question-answer pairs/tuples or a list of sentences/strings.

        This method can take a while if NER_ON is enabled.

        The vocab file's assets are returned as a dictionary and have the
        following attrs:
            'signature': The source file's signature
            'vocab_tokens': a list of vocab tokens decreasingly order
                            by frequency.
            'NER_tokens': (optional) list of NER tokens.
            'NER_label_to_token_dict': (optional) dict mapping a NER
                                       label to token set.

        Note that vocab tokens are all lower cased.

        :param vocab_file: The path to the JSON vocab file with the vocab data.
        :param NER_on: Toggle for Name Entity Recognition features.
        :param verbose: Toggle for update messages.
        :return vocab file's assets as a dictionary.
        """
        count = 0
        tok_freq = {}
        NER_tokens = set()
        ner_label_tokens = set()
        NER_label_to_token_dict = {}
        vocab_data = json.load(open(vocab_file))["data"]

        def process_for_entities(nlp_entities):
            for entity in nlp_entities:
                ner_label_tokens.add(f"<{entity.label_}>")
                for tok in nltk.word_tokenize(entity.text):
                    NER_tokens.add(tok)
                    label = f"<{entity.label_}>"
                    if label in NER_label_to_token_dict:
                        NER_label_to_token_dict[label].add(tok)
                    else:
                        NER_label_to_token_dict[label] = {tok}

        def process_for_freq(tokens):
            for tok in tokens:
                if tok not in NER_tokens:
                    tok = tok.lower()
                    if tok in tok_freq:
                        tok_freq[tok] += 1
                    else:
                        tok_freq[tok] = 1

        def progress_message():
            nonlocal count
            if verbose:
                count += 1
                sys.stdout.write(f"\rParsed {count}/{len(vocab_data)} lines of vocab data.")
                sys.stdout.flush()

        if len(vocab_data) == 0:
            raise ValueError(f"{vocab_file} contains no data.")

        print(f">> Creating vocab assets for '{vocab_file}' ({len(vocab_data)} lines of data) <<")

        first_vocab_el = vocab_data[0]
        if type(first_vocab_el) == str:
            for line in vocab_data:
                if NER_on:
                    process_for_entities(NER_NLP(line).ents)
                process_for_freq(nltk.word_tokenize(line))
                progress_message()
        elif (type(first_vocab_el) == list or type(first_vocab_el) == tuple) and len(first_vocab_el) == 2:
            for question, answer in vocab_data:
                if NER_on:
                    process_for_entities(itertools.chain(NER_NLP(question).ents, NER_NLP(answer).ents))
                process_for_freq(itertools.chain(nltk.word_tokenize(question), nltk.word_tokenize(answer)))
                progress_message()
        else:
            raise ValueError(f"Vocab data: '{vocab_data}' is not supported.")

        vocab_tokens = sorted(list(tok_freq.keys()), key=lambda w: tok_freq.get(w, 0), reverse=True)

        vocab_assets = {
            'signature': f"{vocab_file} (last_mod: {os.path.getmtime(vocab_file)})",
            'vocab_tokens': vocab_tokens,
            'NER_tokens': NER_tokens,
            'NER_label_to_token_dict': NER_label_to_token_dict
        }
        return vocab_assets

    def cache_id(self):
        """
        Returns an integer that depends on the last modified times
        of the files for this instance's cache.
        """
        lst = [os.path.getmtime(f"{self._cache_dir}/f")
               for f in os.listdir(self._cache_dir) if f[-7:] == '.pickle']
        return sum(lst)

    def _create_and_cache_vocab(self, vocab_size, verbose):
        """
        Private method used in the constructor.

        Creates the vocab data dictionary from the vocab file's assets.

        The script attempts to load the desired vocab asset from the cache.
        If none is found, the vocab file's assets is created and cached.
        """
        if not os.path.exists(f"{self._cache_dir}/vocab_assets"):
            os.makedirs(f"{self._cache_dir}/vocab_assets", exist_ok=True)

        vocab_asset_cache_file = f"{self._cache_dir}/vocab_assets/{self.vocab_file_sig}.pickle"
        try:
            if self.ignore_cache:
                raise ValueError("Ignoring cache.")
            vocab_assets = pickle.load(open(vocab_asset_cache_file, 'rb'))
            if vocab_assets['signature'] != self.vocab_file_sig:
                raise ValueError("Cached vocab asset signatures do not match.")
        except (FileNotFoundError, ValueError, KeyError) as e:
            if verbose:
                print(f"Exception encountered when loading vocab assets: {type(e).__name__}, {e}")
            vocab_assets = ChatBot.create_vocab_asset(self.vocab_file, self.ner_enabled, verbose=verbose)
            pickle.dump(vocab_assets, open(vocab_asset_cache_file, 'wb'))

        # Hardcoded special tokens. DO NOT change the order of PADD, START and UNK.
        special_tokens = ["<PADD>", "<START>", "<UNK>"] + list(vocab_assets['NER_label_to_token_dict'].keys())
        vocab_tokens = vocab_assets['vocab_tokens']
        if vocab_size is not None:
            vocab_tokens = vocab_tokens[:vocab_size - len(special_tokens)]
        np.random.shuffle(vocab_tokens)
        vocab = special_tokens + vocab_tokens

        dump = {"signature": vocab_assets['signature'],
                "vocab_size": vocab_size,
                "token_to_id": {c: i for i, c in enumerate(vocab)},
                "id_to_token": {i: c for i, c in enumerate(vocab)},
                "NER_tokens": vocab_assets['NER_tokens'],
                "NER_label_to_token_dict": vocab_assets['NER_label_to_token_dict']}

        pickle.dump(dump, open(f"{self._cache_dir}/vocab.pickle", 'wb'))
        self.vocab_data_dict = dump
        if verbose:
            print(f"Cached vocab file. Vocab size = {vocab_size}, Vocab Sig = {self.vocab_file_sig}")
        return dump

    def define_models(self, latent_dim):
        """
        Define the encoder and decoder models for this instance.

        :param latent_dim: The inner dimensionality of the Encoder and Decoder's LSTM.
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
        :return: The vocab encoding of the sentence as described above.
        """
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
        Private method for the 'train' and decoder/predictor methods.

        This creates the true one hot encoded tensor from the VOCAB_ENCODED_VECTOR.

        A vocab encoded vector is a vector where element i is the token index
        (from the token_to_id dict of this obj) of token i of the sentence
        that said vector is encoding.

        :param vocab_encoded_vector: The encoded vector described above.
        """
        if len(vocab_encoded_vector) > 0:
            shape = (len(vocab_encoded_vector), len(vocab_encoded_vector[0]), len(self.token_to_id_dict))
            encoding = np.zeros(shape)
            for i, seq in enumerate(vocab_encoded_vector):
                for j, tok_id in enumerate(seq):
                    encoding[i, j, tok_id] = 1
            return encoding

    def _create_and_cache_vocab_encoding(self, training_data_pairs, verbose):
        """
        Private method used in the 'train' & 'batch_generator' methods of this class.

        This method creates and caches 3 matrices of 'vocab encoded' training data
        (training data is from TRAINING_DATA_PAIRS) for the training data generator.

        :param training_data_pairs: A list of question answer pairs as training data.
        :param verbose: Toggle update messages
        """

        def is_valid_data(question, answer):
            q_toks, a_toks = nltk.word_tokenize(question), nltk.word_tokenize(answer)
            if self.train_data_filter_mode == 1 and '?' not in q_toks:
                return False
            if self.train_data_filter_mode == 2 and not ('?' in q_toks and '?' in a_toks):
                return False
            return len(q_toks) <= self.n_in and len(a_toks) <= self.n_out

        encoded_x1, encoded_x2, encoded_y = [], [], []
        trained_QA_pairs = []

        print(">> Vocab Encoding Training Data <<")
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
                sys.stdout.write(f"\rProcessed {i + 1}/{len(training_data_pairs)} Question-Answer Pairs.")
                sys.stdout.flush()

        self._v_encoded_x1, self._v_encoded_x2, self._v_encoded_y = encoded_x1, encoded_x2, encoded_y
        self._trained_QA_pairs = trained_QA_pairs

        pickle.dump({"signature": self.train_data_file_sig, "filter": self.train_data_filter_mode,
                     "data": encoded_x1}, open(f"{self._cache_dir}/x1.pickle", 'wb'))
        pickle.dump({"signature": self.train_data_file_sig, "filter": self.train_data_filter_mode,
                     "data": encoded_x2}, open(f"{self._cache_dir}/x2.pickle", 'wb'))
        pickle.dump({"signature": self.train_data_file_sig, "filter": self.train_data_filter_mode,
                     "data": encoded_y}, open(f"{self._cache_dir}/y.pickle", 'wb'))
        pickle.dump({"signature": self.train_data_file_sig, "filter": self.train_data_filter_mode,
                     "data": trained_QA_pairs}, open(f"{self._cache_dir}/trained_QA_pairs.pickle", 'wb'))
        if verbose:
            print("\nCached vocab encoded training data.")
        return True

    def batch_generator(self, batch_size=32):
        """
        A generator that generates 3 tensors of one-hot encoded sentences
        from the training data at each yield. X1 are for the questions,
        Y are for the answers and X2 are for 1 time step forward shifted answers.
        The BATCH_SIZE dictates the number of 'encoded' sentences in a yielded
        tensor (effectively the length of said tensor).

        Each batch is created from a randomized selection of un-yielded
        training data.

        IMPORTANT:
        This relies on the private method: '_create_and_cache_vocab_encoding'
        being called once before this generator's call as it requires the 3
        vocab encoded training data matrices that said method generates.

        :param batch_size: The number of sentences in each batch yielded.
        """
        if not self._v_encoded_x1 or not self._v_encoded_x2 or not self._v_encoded_y:
            raise RuntimeError("Attempted to generate one-hot encodings without vocab encoded training data.")

        lst = list(range(len(self._v_encoded_y)))
        np.random.shuffle(lst)
        queue = deque(lst)
        batch_num = 0
        total_batch_count = int(np.ceil(len(lst) / batch_size))

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
        Check if the current instance has vocab encodings that uses
        this instance's vocab.

        An encoding is valid if this instance's vocab was used for this
        instance's encodings.

        The validation uses an idea similar to the bloom filter.
        The 'hash' portion of this bloom filter idea is solely
        dependent on numpy's shuffle of the vocab.

        Note that there is a slim chance of a false positive.
        This chance can be lowered by increasing N.
        """
        if not self._v_encoded_x1 or not self._v_encoded_x2 \
                or not self._v_encoded_y or not self._trained_QA_pairs:
            return False
        if not (len(self._v_encoded_x1) == len(self._v_encoded_x2) == len(self._v_encoded_y)):
            return False

        N = 30
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

    def _load_cached_v_encoded_train_data(self, verbose):
        """
        Private method to load the encoded training data from cache.

        This does not load the cache file if the signature or
        filter does not match this instance's data file or filter
        setting (respectively).
        """
        if not self.ignore_cache and os.path.isfile(f"{self._cache_dir}/x1.pickle") \
                and os.path.isfile(f"{self._cache_dir}/x2.pickle") \
                and os.path.isfile(f"{self._cache_dir}/y.pickle") \
                and os.path.isfile(f"{self._cache_dir}/trained_QA_pairs.pickle"):
            x1_file_dict = pickle.load(open(f"{self._cache_dir}/x1.pickle", 'rb'))
            x2_file_dict = pickle.load(open(f"{self._cache_dir}/x2.pickle", 'rb'))
            y_file_dict = pickle.load(open(f"{self._cache_dir}/y.pickle", 'rb'))
            QA_file_dict = pickle.load(open(f"{self._cache_dir}/trained_QA_pairs.pickle", 'rb'))

            try:
                if not (self.train_data_file_sig == x1_file_dict["signature"] == x2_file_dict["signature"]
                        == x2_file_dict["signature"] == y_file_dict["signature"] == QA_file_dict["signature"]):
                    raise ValueError("Cached training data's signature does not match.")
                if not (self.train_data_filter_mode == x1_file_dict['filter'] == x2_file_dict['filter']
                        == x2_file_dict['filter'] == y_file_dict['filter'] == QA_file_dict['filter']):
                    raise ValueError("Cached training data's filter does not match.")
                self._v_encoded_x1 = x1_file_dict['data']
                self._v_encoded_x2 = x2_file_dict['data']
                self._v_encoded_y = y_file_dict['data']
                self._trained_QA_pairs = QA_file_dict['data']
                return True
            except (ValueError, AttributeError, TypeError, KeyError) as e:
                if verbose:
                    print(f"Exception when loading cached training data: {type(e).__name__}, {e}")
        return False

    def train(self, data_file, filter_mode, latent_dim, epoch, batch_size, split_pct, verbose):
        """
        Trains this instances encoder and decoder LSTMs (= the Seq2Seq model).

        Note that this method will define a new model for this instance if it either
        doesn't have model or if the latent dimension do not match. Otherwise it will
        'keep training' the model for this instance.

        Note that DATA_FILE is expected to come as a json file where said
        file is has a list of question-answer pairs saved on attribute: 'data'.
            Said list has the following form:
                [...,["Did you change your hair?", "No."], ["Hi!", "Hello."],...]

        Filter Modes:
            0) Only take Questions that have n_in number of tokens and only take
            Answers that have n_out number of tokens.
            1) Mode 0 AND Question must have a '?' token.
            2) Mode 0 AND Question & Answer must have a '?' token.

        :param data_file: The path to the json file containing the question-answer pairs.
        :param filter_mode: An int that determines the filter mode of the training data.
        :param latent_dim: The dimensionality of the Encoder and Decoder's LSTM.
        :param epoch: number of epochs in training.
        :param batch_size: size of the batch in training.
        :param split_pct: a float between 0 and 1. It is the percentage of training data
                          that is held out for validation.
        :param verbose: Toggle update messages.
        """
        self.train_data_file_sig = f"{data_file} (last_mod: {os.path.getmtime(data_file)})"
        self.train_data_file = data_file
        self.train_data_filter_mode = filter_mode

        # Define model if none is defined already.
        if self.model is None or self.decoder is None or self.encoder is None \
                or (self.model.layers and self.model.layers[-1].input_shape[2] != latent_dim):
            self.define_models(latent_dim)
            if verbose:
                print(self.model.summary())
                print("Defined new model.\n")
        elif verbose:
            print(self.model.summary())
            print("[!] Using a pre-defined (and possibly pre-trained) model.\n")

        # Load cached training data encodings.
        self._load_cached_v_encoded_train_data(verbose=verbose)
        if not self.has_valid_vocab_encodings():
            data_pairs = json.load(open(self.train_data_file))["data"]
            self._create_and_cache_vocab_encoding(training_data_pairs=data_pairs, verbose=verbose)
        elif verbose:
            print("[!] Using cached training data encodings.")

        self.train_cache_id = self.cache_id()

        try:  # Recover model if possible, otherwise setup recovering variables.
            if self.ignore_cache:
                raise ValueError("Ignoring cache")
            recovery_sig = pickle.load(open(f'{self._cache_dir}/temp_model/sig.pickle', 'rb'))
            if recovery_sig['OPTIONS'] == OPTIONS and recovery_sig['cache_id'] == self.train_cache_id:
                self.encoder.load_weights(f'{self._cache_dir}/temp_model/encoder.h5')
                self.decoder.load_weights(f'{self._cache_dir}/temp_model/decoder.h5')
            i = recovery_sig['epoch_count']
            if verbose:
                print("[!] Recovered model from cache.")
        except (FileNotFoundError, KeyError, ValueError):
            os.makedirs(f'{self._cache_dir}/temp_model', exist_ok=True)
            i = 0

        if epoch > 0:
            print(f">> Training on {len(self._v_encoded_y)} Question-Answer pairs <<")

        # Train the model.
        while i < epoch:
            for X_1, X_2, Y, batch_counter in self.batch_generator(batch_size=batch_size):
                if verbose:
                    print(f"\rEpoch: {i+1}/{epoch}, Batch: {batch_counter}. \tTraining...")
                X_1t, X_2t, Y_t, X_1v, X_2v, Y_v = ChatBot._create_validation_split(X_1, X_2, Y, split_pct)
                self.model.fit([X_1t, X_2t], Y_t, epochs=1, batch_size=batch_size,
                               validation_data=([X_1v, X_2v], Y_v), verbose=verbose)

            sys.stdout.write('\x1b[2K')
            sys.stdout.flush()
            sys.stdout.write(f"\rFinished epoch: {i+1}/{epoch}")
            sys.stdout.flush()

            pickle.dump({'OPTIONS': OPTIONS, 'epoch_count': i, 'cache_id': self.train_cache_id},
                        open(f'{self._cache_dir}/temp_model/sig.pickle', 'wb'))
            self.encoder.save_weights(f'{self._cache_dir}/temp_model/encoder.h5')
            self.decoder.save_weights(f'{self._cache_dir}/temp_model/decoder.h5')
            i += 1

        if verbose:
            print(f"\nTraining Complete.\nTrained on {len(self._v_encoded_y)} Question-Answer pairs")
        shutil.rmtree(f'{self._cache_dir}/temp_model')
        return True

    def _predict(self, X_in):
        """
        Private method used in the 'chat' method.

        This predicts a response (aka decoding) based on the one-hot
        encoded sentence X_IN.
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

    def _vector_to_sentence(self, vector):
        """
        Private method used in the 'chat' method.

        Converts a (predicted) one-hot encoded VECTOR to a sentence.
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
        """
        Main chat loop of the chat bot.
        """
        if not self:
            raise RuntimeError("Attempted to chat with an untrained model.")
        print("Chat Bot ready, type anything to start: (Ctrl + C or type '!EXIT' to stop chatting)")
        try:
            while True:
                sys.stdout.write(">")
                sys.stdout.flush()

                input_str = input()
                if input_str == '!EXIT':
                    print("Done Chatting...\n")
                    return True

                vocab_encoded_X_in = [self.v_encode(input_str, self.n_in)]
                X_in = self._one_hot_encode(vocab_encoded_X_in)
                Y_hat = self._predict(X_in)
                print("Response: {}".format(self._vector_to_sentence(Y_hat)))
                print(" ")
        except KeyboardInterrupt:
            print("\nDone Chatting...\n")
            return True

    def save(self, directory):
        """
        :param directory: The directory of where this instance is going to be saved.
        """
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        else:
            shutil.rmtree(directory)
            os.mkdir(directory)

        pickle.dump(OPTIONS, open(f"{directory}/options.pickle", 'wb'))
        self.encoder.save_weights(f"{directory}/encoder.h5")
        self.decoder.save_weights(f"{directory}/decoder.h5")
        shutil.copyfile(self.train_data_file, f"{directory}/[T-DAT]{self.train_data_file}")
        shutil.copyfile(self.vocab_file, f"{directory}/[V-DAT]{self.vocab_file}")

        if not os.path.exists(f"{directory}/{self._cache_dir}"):
            os.mkdir(f"{directory}/{self._cache_dir}")
        for file in filter(lambda f: f[-7:] == '.pickle', os.listdir(self._cache_dir)):
            shutil.copyfile(f'{self._cache_dir}/{file}', f"{directory}/{self._cache_dir}/{file}")
        print(f"\nSaved the trained model to: '{directory}'.")
        return True


def get_options():
    opts = ArgumentParser()
    opts.add_argument('-i', '--n_in', dest='n_in', type=int, default=10,
                      help="The number of time steps for the encoder. Default = 10.")
    opts.add_argument('-o', '--n_out', dest='n_out', type=int, default=20,
                      help="The number of time setps for the decoder. Default = 20.")
    opts.add_argument('-l', '--latent_dim', dest='latent_dim', type=int, default=128,
                      help="The inner dimensionality of the Encoder and Decoder's LSTM. Default = 128.")
    opts.add_argument('-v', '--vocab_size', dest='vocab_size', type=int, default=None,
                      help='The size of the vocab of the Chatbot. Default = None')
    opts.add_argument('-f', '--vocab_file', dest='vocab_file', type=str, default=None,
                      help="The directory of the JSON file that is used to define the vocab. "
                           "The 'data' attribute can be either question-answer pairs or just strings/sentences. "
                           "Default = whatever the train_file is.")
    opts.add_argument("-I", '--ignore_cache', action="store_true", dest="ignore_cache",
                      help="Forces the script to ignore the cached files.")
    opts.add_argument("-N", '--NER_disable', action="store_true", dest="NER_disable",
                      help="Turns off Name Entity Recognition as part of the chatbot model. "
                           "Note that NER adds a considerable amount of complexity in encoding "
                           "the training data.")
    opts.add_argument("-M", '--verbose', action="store_true", dest="verbose",
                      help="Toggles verbose on.")
    opts.add_argument('-t', '--train_file', dest='train_file', type=str, default="Cornell_Movie_Dialogs_Data.json",
                      help="The directory of the JSON file that is used to train the model. "
                           "The 'data' attribute must be a list of question-answer pairs."
                           "Default = 'Cornell_Movie_Dialogs_Data.json'")
    opts.add_argument('-c', '--filter_mode', dest='filter_mode', type=int, default=0,
                      help="An integer that dictates the filter imposed of the data. MODES: {0, 1, 2}. "
                           "Mode 0: Only take Questions that have N_in number of tokens and only take Answers "
                           "that have n_out number of tokens. Mode 1: All of Mode 0 AND Questions must have "
                           "a '?' token. Mode 2: All of Mode 0 AND Question & Answer must have a '?' token. "
                           "Default = 0")
    opts.add_argument('-e', '--epoch', dest='epoch', type=int, default=500,
                      help="The number of epochs for training. Default = 100.")
    opts.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=32,
                      help="The batch size for training. Default = 32.")
    opts.add_argument('-s', '--split', dest='split', type=float, default=0.35,
                      help="The percentage (float between 0 - 1) of data held out for validation. "
                           "Default = 0.35")
    opts.add_argument('-m', '--saved_models_dir', dest='saved_models_dir', type=str, default="saved_models",
                      help="The directory for all of the saved (trained) models. "
                           "Default = 'saved_models'")
    options = opts.parse_args()
    if options.vocab_file is None:
        options.vocab_file = options.train_file
    return options


def get_saved_model(directory):
    if not os.path.exists(directory):
        print(f"'{directory}' is an invalid directory.")
        return None

    saved_models = [f for f in os.listdir(directory) if f != '.DS_Store']
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


OPTIONS = get_options()  # Global options for interactive sessions.

if __name__ == "__main__":
    # TODO: IMPLEMENTED AND TEST DECODER NER FEATURES.
    # TODO: publish README...

    if not os.path.exists(OPTIONS.saved_models_dir):
        os.makedirs(OPTIONS.saved_models_dir, exist_ok=True)

    sys.stdout.write("\rLoad a saved model? (y/n) ")
    sys.stdout.flush()
    user_input = input()

    if user_input[0].lower() == 'y':
        saved_model_dir = f"{OPTIONS.saved_models_dir}/{get_saved_model(OPTIONS.saved_models_dir)}"
    else:
        saved_model_dir = None
    try:
        if saved_model_dir is None:
            raise ValueError("No file to load.")
        loaded_model_options = pickle.load(open(f"{saved_model_dir}/options.pickle", 'rb'))
        chat_bot = ChatBot.load(saved_model_dir, loaded_model_options)
        print(f"\nLoaded model: {saved_model_dir.split('/')[-1]}")
    except (ValueError, FileNotFoundError):
        new_model_name = None

        sys.stdout.write("\rSave the newly trained model? (y/n) ")
        sys.stdout.flush()
        user_input = input()

        if user_input[0].lower() == 'y':
            sys.stdout.write("\r(Required) Name of newly trained model? ")
            sys.stdout.flush()
            new_model_name = input()

        chat_bot = ChatBot(OPTIONS.n_in, OPTIONS.n_out, OPTIONS.vocab_size, OPTIONS.vocab_file,
                           OPTIONS.ignore_cache, not OPTIONS.NER_disable, OPTIONS.verbose)
        chat_bot.train(OPTIONS.train_file, OPTIONS.filter_mode, OPTIONS.latent_dim, OPTIONS.epoch,
                       OPTIONS.batch_size, OPTIONS.split, OPTIONS.verbose)

        if new_model_name:
            new_model_directory = f"{OPTIONS.saved_models_dir}/{new_model_name}"
            chat_bot.save(new_model_directory)
    chat_bot.chat()
