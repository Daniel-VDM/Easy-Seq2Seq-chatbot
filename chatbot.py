import numpy as np
import nltk
import os
import sys
import pickle
import itertools
import spacy
import keras
from collections import deque
from nltk.corpus import stopwords
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.preprocessing import sequence

# Globals (can be modified)
TRAIN_FIRST = False
N_in = 10
N_out = 20
EPOCHS = 100
SENTENCE_LEN_LIM = 15  # in token counts.
MAX_VOCAB_SIZE = 10000
VALIDATION_SPLIT = 0.35
BATCH_SIZE = 128

# TODO: || it to train bigger models quicker... also maybe do some cleanup to make it ezer to run.

# Do not modify.
WORD_TO_ID_DICT = {}
ID_TO_WORD_DICT = {}
ENTITY_WORDS = pickle.load(open("ENTITY_WORDS.pickle", 'rb'))
NLP = spacy.load('en')


def define_models(n_input, n_output, n_units):
    """
    seq-to-seq encoder/decoder function. Found on most seq-to-seq tutorials.
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


def one_hot_encode_array(iterable, n, vocab_len):
    encoding = np.zeros((len(iterable), n, vocab_len))
    for i, seq in enumerate(iterable):
        for j, index in enumerate(seq):
            encoding[i, j, index] = 1
    return encoding


def one_hot_encode_list(iterable, vocab_len):
    lst = []
    for seq in iterable:
        time_steps = []
        for index in seq:
            vector = [0 for _ in range(vocab_len)]
            vector[index] = 1
            time_steps.append(vector)
        lst.append(time_steps)
    return lst


def create_vocab_file():
    word_freq = {}

    data_1 = open("dataset.txt", "r").read().split("\n\n")
    data_2 = open("movie_lines_filtered.tsv", encoding='utf-8', errors='ignore').read().split('\n')
    i, total_length = 0, len(data_1) + len(data_2)
    for line in data_1:
        if not line:
            continue
        for word in nltk.word_tokenize(line):
            if not word:
                continue  # Blank line error handle
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
        i += 1
        sys.stdout.write(f"\rCreating Vocab, parsing {i}/{total_length} document lines.")
        sys.stdout.flush()

    for line in data_2:
        if not line:
            continue
        for word in nltk.word_tokenize(line.split("\t")[4]):
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
        i += 1
        sys.stdout.write(f"\rCreating Vocab, parsing {i}/{total_length} document lines.")
        sys.stdout.flush()

    common_words = set(filter(lambda w: w.isalpha(), word_freq.keys()))
    vocab_words = (common_words - ENTITY_WORDS)
    vocab_words = sorted(vocab_words, key=lambda w: word_freq[w], reverse=True)

    ner_tokens = pickle.load(open("NER_TAGS_OF_DATA.pickle", 'rb'))
    special_tokens = ["<PADD>", "<START>", "<UNK>"] + ner_tokens

    vocab = list(stopwords.words('english')) + list(map(lambda w: w.lower(), vocab_words))
    vocab = set(vocab[:MAX_VOCAB_SIZE])

    dump = {"word_to_id": dict((c, i) for i, c in enumerate(itertools.chain(special_tokens, vocab))),
            "id_to_word": dict((i, c) for i, c in enumerate(itertools.chain(special_tokens, vocab)))}
    print("\nPickled vocab file.")
    pickle.dump(dump, open("vocab.pickle", 'wb'))
    return dump


def vectorize(sentence, unk_token_id=None):
    sentence = sentence.strip()
    unk_token_id = unk_token_id if unk_token_id else WORD_TO_ID_DICT["<UNK>"]
    sentence_tokens = np.array(list(filter(lambda s: s.isalpha(), nltk.word_tokenize(sentence))))
    vector = np.zeros(len(sentence_tokens), dtype=int)

    entity = {}
    if any(x in ENTITY_WORDS for x in set(sentence_tokens)):
        for ent in NLP(sentence).ents:
            for w in nltk.word_tokenize(ent.text):
                entity[w] = f"<{ent.label_}>"

    for i, word in enumerate(sentence_tokens):
        if word in entity:
            word_id = WORD_TO_ID_DICT[entity[word]]
        else:
            word_id = WORD_TO_ID_DICT.get(word.lower(), unk_token_id)
        vector[i] = word_id
    return vector


def training_vector_generator():
    q_and_a_lst = []

    data = open("movie_lines_filtered.tsv", encoding='utf-8', errors='ignore').read().split('\n')
    data = [x for x in data if x]
    for i in range(len(data) - 1):
        line_a, a_uter, a_mov, _, a_text = data[i].split("\t")[:5]
        line_b, b_uter, b_mov, _, b_text = data[i + 1].split("\t")[:5]
        line_a = int("".join([s for s in line_a if s.isdigit()]))
        line_b = int("".join([s for s in line_b if s.isdigit()]))

        if a_uter != b_uter and a_mov == b_mov and line_b == line_a + 1\
                and len(a_text.split(" ")) <= SENTENCE_LEN_LIM \
                and len(b_text.split(" ")) <= SENTENCE_LEN_LIM:
            q_and_a_lst.append((a_text.strip(), b_text.strip()))

    for line in open("dataset.txt", 'r').read().split("\n\n"):
        question, answer = line.split("\n")
        if len(question.split(" ")) <= SENTENCE_LEN_LIM \
                and len(answer.split(" ")) <= SENTENCE_LEN_LIM:
            q_and_a_lst.append((question.strip(), answer.strip()))

    batch_number, doc_number = 0, 0
    total_batch_count = int(np.ceil(len(q_and_a_lst)/BATCH_SIZE))
    np.random.shuffle(q_and_a_lst)
    queue = deque(q_and_a_lst)

    while queue:
        batch_size = min(len(queue), BATCH_SIZE)
        lst = [queue.pop() for _ in range(batch_size)]

        questions = map(lambda tup: tup[0], lst)
        answers = map(lambda tup: tup[1], lst)

        X_1 = np.empty((batch_size,), dtype=bytearray)
        X_2 = np.empty((batch_size,), dtype=bytearray)
        Y = np.empty((batch_size,), dtype=bytearray)

        # Create training vectors from read data.
        for index, question in enumerate(questions):
            X_1[index] = vectorize(question)

            doc_number += 1
            sys.stdout.write(f"\rVectorizing Training data {doc_number}/{2*len(q_and_a_lst)}")
            sys.stdout.flush()

        for index, answer in enumerate(answers):
            vector = vectorize(answer)
            Y[index] = vector
            vector = [WORD_TO_ID_DICT["<START>"]] + list(vector)[:-1]
            X_2[index] = np.array(vector)

            doc_number += 1
            sys.stdout.write(f"\rVectorizing Training data {doc_number}/{2*len(q_and_a_lst)}")
            sys.stdout.flush()

        X_1 = sequence.pad_sequences(X_1, maxlen=N_in, padding='post')
        X_2 = sequence.pad_sequences(X_2, maxlen=N_out, padding='post')
        Y = sequence.pad_sequences(Y, maxlen=N_out, padding='post')

        encode_len = len(WORD_TO_ID_DICT)
        X_1 = one_hot_encode_array(X_1, N_in, encode_len)
        X_2 = one_hot_encode_array(X_2, N_out, encode_len)
        Y = one_hot_encode_array(Y, N_out, encode_len)

        batch_number += 1

        yield X_1, X_2, Y, f"{batch_number}/{total_batch_count}"


def create_validation_split(X_1, X_2, Y, percentage=None):
    percentage = percentage if percentage else VALIDATION_SPLIT
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


def train():
    model, encoder, decoder = define_models(len(WORD_TO_ID_DICT), len(WORD_TO_ID_DICT), 128)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    print(model.summary())
    print("\n\n-==TRAINING=--\n")
    for ep in range(EPOCHS):
        for X_1, X_2, Y, batch_counter in training_vector_generator():
            sys.stdout.write('\x1b[2K')
            print(f"\rEpoch: {ep}/{EPOCHS}, Batch: {batch_counter}. \tTraining...")
            X_1t, X_2t, Y_t, X_1v, X_2v, Y_v = create_validation_split(X_1, X_2, Y)
            sys.stdout.flush()
            model.fit([X_1t, X_2t], Y_t, epochs=1,
                      batch_size=BATCH_SIZE, validation_data=([X_1v, X_2v], Y_v))

    print("\nDone training, saved model to file.")

    encoder.save_weights("encoding_model.h5")
    decoder.save_weights("decoding_model.h5")


def load_encoder_decoder():
    if TRAIN_FIRST or not os.path.isfile("encoding_model.h5") \
            or not os.path.isfile("decoding_model.h5"):
        train()
    _, encoder, decoder = define_models(len(WORD_TO_ID_DICT), len(WORD_TO_ID_DICT), 128)
    encoder.load_weights("encoding_model.h5")
    decoder.load_weights("decoding_model.h5")
    return encoder, decoder


def process_input(input_line):
    X_in = sequence.pad_sequences([vectorize(input_line)], maxlen=N_in, padding='post')
    return np.array(one_hot_encode_list(X_in, len(WORD_TO_ID_DICT)))


def predict(encoder, decoder, X_in, encode_len):
    curr_in_state = encoder.predict(X_in)
    curr_out_state = [np.array(one_hot_encode_list([[WORD_TO_ID_DICT["<START>"]]], encode_len))]
    Y_hat = []
    for t in range(N_in):
        prediction, h, c = decoder.predict(curr_out_state + curr_in_state)
        curr_in_state = [h, c]
        curr_out_state = [prediction]
        Y_hat.append(prediction)
    return np.array(Y_hat)


def vector_to_words(vector):
    words = []
    for el in vector:
        word_id = np.argmax(el)  # Fetch index that has 1 as element.
        word = ID_TO_WORD_DICT.get(word_id, "<UNK>")
        if word == "<PADD>":
            return " ".join(words)
        words.append(word)
    return " ".join(words)


def main():
    global ID_TO_WORD_DICT, WORD_TO_ID_DICT

    if os.path.isfile("vocab.pickle") and not TRAIN_FIRST:
        vocab_data = pickle.load(open("vocab.pickle", 'rb'))
    else:
        vocab_data = create_vocab_file()

    WORD_TO_ID_DICT, ID_TO_WORD_DICT = vocab_data["word_to_id"], vocab_data["id_to_word"]

    encoder, decoder = load_encoder_decoder()

    print("Chat Bot ready, type anything to start:")
    while True:
        sys.stdout.write(">")
        sys.stdout.flush()
        X_in = process_input(input())
        Y_hat = predict(encoder, decoder, X_in, len(WORD_TO_ID_DICT))
        print("Response: {}".format(vector_to_words(Y_hat)))
        print(" ")


if __name__ == "__main__":
    main()
