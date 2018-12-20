import nltk
import sys

dat = []

data = open("movie_lines.tsv", encoding='utf-8', errors='ignore').read().split('\n')
for i in range(len(data) - 1):
    line_id, char, mov, char_name, text = data[i].split("\t")[:5]
    if not text:
        continue
    clean_id = "".join([s for s in line_id if s.isdigit()])
    if len(nltk.word_tokenize(text)) <= 20:
        dat.append((clean_id, char, mov, char_name, text))
    sys.stdout.write(f"\rFiltered {i}/{len(data)}")
    sys.stdout.flush()
print("Writing...")


dat.sort(key=lambda e: int(e[0]))

with open("movie_lines_filtered.tsv", 'w', encoding='utf-8') as f:
    for line in dat:
        write_line = "\t".join(line)
        f.write(write_line + "\n")