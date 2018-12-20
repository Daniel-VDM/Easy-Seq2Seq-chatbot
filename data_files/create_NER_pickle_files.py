import pickle
import spacy
import sys

ents = set()
word_of_ents = set()
nlp = spacy.load('en')

data = open("movie_lines_filtered.tsv", encoding='utf-8', errors='ignore').read().split('\n')
i = 0
for line in data:
    sentence = line.split("\t")[4]
    doc = nlp(sentence)
    if doc.ents:
        for en in doc.ents:
            ents.add(en.label_)
            word_of_ents.add(en.text)
    i += 1
    sys.stdout.write(f"\r{i}/{len(data)} done.")
    sys.stdout.flush()

print("\nDumped Entity files.")
dump = [f"<{e}>" for e in ents]
pickle.dump(dump, open("NER_TAGS_OF_DATA.pickle", 'wb'))
pickle.dump(word_of_ents, open("ENTITY_WORDS.pickle", 'wb'))
