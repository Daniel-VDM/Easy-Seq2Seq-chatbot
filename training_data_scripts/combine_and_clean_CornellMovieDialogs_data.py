import nltk
import json
import sys
from datetime import datetime
import os

dat = []

sys.stdout.write("What is the Cornell Movie Dialog file name? ")
sys.stdout.flush()
file_name = input()

data = open(file_name, encoding='utf-8', errors='ignore').read().split('\n')

if not data[-1]:
    data.pop()

sys.stdout.write("\rName of this newly generated vocab file? ")
sys.stdout.flush()
out_file_name = input()

line_num = 0
for i in range(len(data) - 1):
    line_id, char, mov, char_name, text = data[i].split("\t")[:5]
    if not text:
        continue
    clean_id = "".join([s for s in line_id if s.isdigit()])
    line_num = max(int(clean_id), line_num)
    if 0 < len(nltk.word_tokenize(text)) <= 20:
        dat.append((clean_id, char, mov, char_name, text))
    sys.stdout.write(f"\rProcessed {i}/{len(data)}")
    sys.stdout.flush()

dat.sort(key=lambda e: int(e[0]))

# Q and A filtering is done below, explained in writeup.
Q_A_pairs = []
Q = []
A = []

for i in range(len(dat) - 1):
    line_a, a_uter, a_mov, _, a_text = dat[i]
    line_b, b_uter, b_mov, _, b_text = dat[i + 1]
    if a_uter != b_uter and a_mov == b_mov and int(line_b) == int(line_a) + 1:
        Q_A_pairs.append((a_text, b_text))
        Q.append(a_text)
        A.append(b_text)

# Hardcoded Data
if os.path.isfile("dataset.txt"):
    new_stuff = open("dataset.txt", "r").read().split("\n\n")
    if not new_stuff[-1]:
        new_stuff.pop()

    for line in new_stuff:
        if not line:
            continue
        question, answer = line.split("\n")
        Q_A_pairs.append((question, answer))
        Q.append(question)
        A.append(answer)

dump = {
    "source_file": file_name,
    "creation_date": str(datetime.utcnow()) + ' UTC',
    "question_answer_pairs": Q_A_pairs,
    "vocab_data": Q_A_pairs,
    "questions": Q,
    "answers": A
}

with open(f"{out_file_name}.json", 'w', encoding='utf-8') as f:
    json.dump(dump, f)
print(f"\nDone. Wrote json file as: '{out_file_name}.json'")
