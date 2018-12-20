import nltk
import sys

dat = []

data = open("movie_lines.tsv", encoding='utf-8', errors='ignore').read().split('\n')

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
print("Writing...")


new_stuff = open("dataset.txt", "r").read().split("\n\n")
for i, line in zip(range(1, len(new_stuff)+1, 2), new_stuff):
    if not line:
        continue
    question, answer = line.split("\n")
    dat.append((str(line_num+i), "TEMP_CHAR_A", "TEMP_MOV_A", "A", question))
    dat.append((str(line_num + i + 1), "TEMP_CHAR_B", "TEMP_MOV_B", "B", answer))

dat.sort(key=lambda e: int(e[0]))

with open("movie_lines_filtered.tsv", 'w', encoding='utf-8') as f:
    for line in dat:
        write_line = "\t".join(line)
        f.write(write_line + "\n")

