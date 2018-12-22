import nltk
import json
import sys
import os

dat = []

sys.stdout.write("Input Cornell Movie Dialogs Raw .tsv file name: ")
sys.stdout.flush()
file_name = input()

data = open(file_name, encoding='utf-8', errors='ignore').read().split('\n')

if not data[-1]:
    data.pop()

line_num = 0
for i in range(len(data) - 1):
    line_id, char, mov, char_name, text = data[i].split("\t")[:5]
    if not text:
        continue
    clean_id = "".join([s for s in line_id if s.isdigit()])
    line_num = max(int(clean_id), line_num)
    if 0 < len(nltk.word_tokenize(text)) <= 20:
        dat.append((clean_id, char, mov, char_name, text))
    sys.stdout.write(f"\rFiltered {i}/{len(data)}")
    sys.stdout.flush()

# Hardcoded Data.
if os.path.isfile("dataset.txt"):
    new_stuff = open("dataset.txt", "r").read().split("\n\n")
    if not new_stuff[-1]:
        new_stuff.pop()

    for i, line in zip(range(1, len(new_stuff)+1, 2), new_stuff):
        if not line:
            continue
        question, answer = line.split("\n")
        dat.append((str(line_num+i), "TEMP_CHAR_A", "TEMP_MOV", "A", question))
        dat.append((str(line_num + i + 1), "TEMP_CHAR_B", "TEMP_MOV", "B", answer))

dat.sort(key=lambda e: int(e[0]))

# Q and A filtering is done below, explained in writeup.
jason_dump = []
for i in range(len(dat) - 1):
    line_a, a_uter, a_mov, _, a_text = dat[i]
    line_b, b_uter, b_mov, _, b_text = dat[i + 1]
    if a_uter != b_uter and a_mov == b_mov and int(line_b) == int(line_a) + 1:
        jason_dump.append((a_text, b_text))

with open("Cornell_Movie_Dialogs_Data.json", 'w', encoding='utf-8') as f:
    json.dump(jason_dump, f)
print("\nDone. Wrote json file as: 'Cornell_Movie_Dialogs_Data.json'")
