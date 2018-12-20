#!/usr/bin/sudo python
import numpy as np
import nltk
import os
import sys
import pickle
import itertools
import spacy
import shutil
from collections import deque
from nltk.corpus import stopwords
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.preprocessing import sequence


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
    ENTITY_WORDS = pickle.load(open("ENTITY_WORDS.pickle", 'rb'))
    NLP = spacy.load('en')

    def __init__(self, n_in, n_out, sentence_limit, vocab_size, vocab_filename, vocab_update=False):
        if os.path.isfile("vocab.pickle") and not vocab_update:
            try:
                data_vocab_dicts = pickle.load(open("vocab.pickle", 'rb'))
                if len(data_vocab_dicts["word_to_id"]) != len(data_vocab_dicts["word_to_id"]):
                    raise ValueError("'vocab.pickle' dictionary lengths do not match.")
                if len(data_vocab_dicts["word_to_id"]) != vocab_size:
                    raise ValueError("'vocab.pickle' vocab size is not {}.".format(vocab_size))
            except Exception as e:
                print("Exception encountered when reading vocab data: {}".format(e))
                data_vocab_dicts = self._create_and_save_vocab(vocab_filename, vocab_size)
        else:
            data_vocab_dicts = self._create_and_save_vocab(vocab_filename, vocab_size)

        self.n_in, self.n_out = n_in, n_out
        self.vocab_size = vocab_size
        self.sentence_limit = sentence_limit
        self.word_to_id_dict = data_vocab_dicts["word_to_id"]
        self.id_to_word_dict = data_vocab_dicts["id_to_word"]
        self.encoder, self.decoder = None, None

        if not os.path.exists("temp"):
            os.makedirs("temp")

        os.makedirs("temp/{}".format(self))

    def __del__(self):
        shutil.rmtree("temp/{}".format(self))

    def __bool__(self):
        return self.encoder is not None and self.decoder is not None

    @staticmethod
    def _create_and_save_vocab(filename, vocab_size):
        """
        Creates and pickle's vocab from the training data. Note that the training
        data is expected to come in the following form:

        5 Column separated by a tab character using the following column (in order):
            <lineID>\t<characterID>\t<movieID>\t<char_name>\t<text>

        Vocab uses most frequent words first when truncating the vocab to
        fit the vocab size.

        # TODO: more robust vocab method to handel plain words instead of just 5 col.

        :param filename: file name of training data used to generate dict.
        :param vocab_size: the fixed size of the vocab.
        """
        word_freq, i = {}, 0
        data = open(filename, encoding='utf-8', errors='ignore').read().split('\n')

        for line in data:
            if not line:
                continue
            for word in nltk.word_tokenize(line.split("\t")[4]):
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1
            i += 1
            sys.stdout.write("\rCreating Vocab, parsing {}/{} document lines.".format(i, len(data)))
            sys.stdout.flush()

        alpha_vocab = set(filter(lambda w: w.isalpha(), word_freq.keys()))
        vocab = set(map(lambda w: w.lower(), alpha_vocab - ChatBot.ENTITY_WORDS))

        ner_tokens = pickle.load(open("NER_TAGS_OF_DATA.pickle", 'rb'))
        special_tokens = ["<PADD>", "<START>", "<UNK>"] + ner_tokens

        vocab = sorted(list(vocab), key=lambda w: word_freq.get(w, 0), reverse=True)
        vocab = vocab[:vocab_size - len(special_tokens)]

        dump = {"word_to_id": {c: i for i, c in enumerate(itertools.chain(special_tokens, vocab))},
                "id_to_word": {i: c for i, c in enumerate(itertools.chain(special_tokens, vocab))}}
        print("\nPickled vocab file.")
        pickle.dump(dump, open("vocab.pickle", 'wb'))
        return dump

    @staticmethod
    def _one_hot_encode_to_list(iterable, vocab_len):
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
    def _one_hot_encode_to_array(iterable, n, vocab_len):
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

    def vectorize(self, sentence):
        """
        Note that this is NOT one-hot encoded. Instead, it returns a vector where
        each entry is a word ID, and said entry corresponds to token index of sentence.

        :param sentence: A string that is to be vectorized.
        :return: an encoding/vector (using this objects vocab) of the sentence.
        """
        sentence = sentence.strip()
        unk_token_id = self.word_to_id_dict["<UNK>"]
        sentence_tokens = np.array(list(filter(lambda s: s.isalpha(), nltk.word_tokenize(sentence))))
        vector = np.zeros(len(sentence_tokens), dtype=int)

        entity = {}
        if any(x in ChatBot.ENTITY_WORDS for x in set(sentence_tokens)):
            for ent in ChatBot.NLP(sentence).ents:
                for w in nltk.word_tokenize(ent.text):
                    entity[w] = "<{}>".format(ent.label_)

        for i, word in enumerate(sentence_tokens):
            if word in entity:
                word_id = self.word_to_id_dict[entity[word]]
            else:
                word_id = self.word_to_id_dict.get(word.lower(), unk_token_id)
            vector[i] = word_id
        return vector

    def create_training_generator(self, data_file, batch_size=32):
        """
        A generator that yields batches of one-hot encoded sentences for
        the Seq2Seq model.

        The training data (DATA_FILE) is expected to come in the following form:

        5 Column separated by a tab character using the following column (in order):
            <lineID>\t<characterID>\t<movieID>\t<char_name>\t<text>

        Also, one-hot encoding comes from the vocab file (vocab.pickle).

        :param data_file: The data file in the form described above.
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
                    and len(a_text.split(" ")) <= self.sentence_limit \
                    and len(b_text.split(" ")) <= self.sentence_limit:
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
            X_1 = self._one_hot_encode_to_array(X_1, self.n_in, encode_len)
            X_2 = self._one_hot_encode_to_array(X_2, self.n_out, encode_len)
            Y = self._one_hot_encode_to_array(Y, self.n_out, encode_len)

            batch_number += 1

            yield X_1, X_2, Y, "{}/{}".format(batch_number, total_batch_count)

    def train(self, data_filename, epoch, batch_size=32, split_percentage=0.35, verbose=False):
        """
        Trains the chatbot's seq2seq encoder and decoder LSTMs.

        The training data (DATA_FILENAME) is expected to come in the following form:

        5 Column separated by a tab character using the following column (in order):
            <lineID>\t<characterID>\t<movieID>\t<char_name>\t<text>

        :param data_filename: the data being train on. Must follow format above.
        :param epoch: number of epochs in training.
        :param batch_size: size of the batch in training.
        :param split_percentage: value between 0 and 1, percentage of training
            data held out for validation.
        :param verbose: update messages during training.
        :return:
        """
        model, encoder, decoder = define_models(len(self.word_to_id_dict), len(self.id_to_word_dict), 128)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        if verbose:
            print(model.summary())
        print("\n\n-==TRAINING=--\n")
        for ep in range(epoch):
            for X_1, X_2, Y, batch_counter in self.create_training_generator(data_filename, batch_size):
                if verbose:
                    sys.stdout.flush()
                    sys.stdout.write('\x1b[2K')
                    print("\rEpoch: {}/{}, Batch: {}. \tTraining...".format(ep, epoch, batch_counter))
                X_1t, X_2t, Y_t, X_1v, X_2v, Y_v = self._create_validation_split(X_1, X_2, Y, split_percentage)
                model.fit([X_1t, X_2t], Y_t, epochs=1, batch_size=batch_size, validation_data=([X_1v, X_2v], Y_v))
        print("\nDone training, saved model to file.")
        encoder.save_weights("encoding_model.h5")
        decoder.save_weights("decoding_model.h5")
        # TODO: Backup and save weights and vocab.


# def load_encoder_decoder():
#     if TRAIN_FIRST or not os.path.isfile("encoding_model.h5") \
#             or not os.path.isfile("decoding_model.h5"):
#         train()
#     _, encoder, decoder = define_models(len(WORD_TO_ID_DICT), len(WORD_TO_ID_DICT), 128)
#     encoder.load_weights("encoding_model.h5")
#     decoder.load_weights("decoding_model.h5")
#     return encoder, decoder
#
#
# def process_input(input_line):
#     X_in = sequence.pad_sequences([vectorize(input_line)], maxlen=N_in, padding='post')
#     return np.array(one_hot_encode_to_list(X_in, len(WORD_TO_ID_DICT)))
#
#
# def predict(encoder, decoder, X_in, encode_len):
#     curr_in_state = encoder.predict(X_in)
#     curr_out_state = [np.array(one_hot_encode_to_list([[WORD_TO_ID_DICT["<START>"]]], encode_len))]
#     Y_hat = []
#     for t in range(N_in):
#         prediction, h, c = decoder.predict(curr_out_state + curr_in_state)
#         curr_in_state = [h, c]
#         curr_out_state = [prediction]
#         Y_hat.append(prediction)
#     return np.array(Y_hat)
#
#
# def vector_to_words(vector):
#     words = []
#     for el in vector:
#         word_id = np.argmax(el)  # Fetch index that has 1 as element.
#         word = ID_TO_WORD_DICT.get(word_id, "<UNK>")
#         if word == "<PADD>":
#             return " ".join(words)
#         words.append(word)
#     return " ".join(words)


def main():
    # TODO: ARGPARCER
    # TODO: DECODER & EXIT
    # TODO: TEST ON SIMPLE DEV DATA
    # TODO: FILE WRITE / READ BATCH GENERATOR.
    # TODO: Documentation
    # TODO: First Time installer...
    # TODO: publish README...
    chat_bot = ChatBot(10, 20, 15, 9000, "movie_lines_filtered.tsv")
    chat_bot.train("movie_lines_filtered.tsv", 100, verbose=True)


if __name__ == "__main__":
    main()
