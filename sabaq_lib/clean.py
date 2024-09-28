from difflib import SequenceMatcher
import unicodedata

import pymongo


import spacy

def init_db():
    client = pymongo.MongoClient('mongodb://localhost:27017/')
    db = client['sabaq']
    return db

def init_nlp():
    return spacy.load('fr_dep_news_trf')


# TODO:: 'A la queue leu leu' is not being found in one of the examples
# root cause: beautiful soup is parsing <br> as '' instead of whitespace
# important because this bug immediately spoils all poetic idioms or othre
# idioms that have verse breaks 
if __name__ == '__main__':
    nlp = init_nlp()
    db = init_db()

    print('tokenizing idioms' )

    # Map to a list of idioms
    idioms = map(lambda x: nlp(unicodedata.normalize('NFKC', x['idiom'])), 
                 db['fr_wiktionary'].find({}))

    # Lemmatize the idioms
    # NOTE: We leave stop words, since they are often
    #       important in idioms
    idioms = map(
            lambda doc: [x.lemma_ for x in doc if not x.is_punct],
            idioms)

    # Map to a list of examples
    exs = map(
            lambda x: x['examples'],
            db['fr_wiktionary'].find({}))

    # Flatten the list
    exs = map(
            lambda ex: nlp(ex), 
            (unicodedata.normalize('NFKC', x) for exss in exs for x in exss))

    # Lemmatize the examples
    # NOTE: Make it a tuple, so we can use it as a key
    exs = list(map(
            lambda doc: tuple([x.lemma_ for x in doc if not x.is_punct]), 
            exs))

    key = {}

    # if 'Coligny' not in ' '.join(ex):
    #     continue
        # if 'donner le brebis' not in ' '.join(idiom):
        #     continue
    for idiom in idioms:
        print('idiom', idiom)

        for i, ex in enumerate(exs):

            tag = [0] * len(ex)
            
            for i in range(len(ex) - len(idiom) + 1):
                cur = ex[i:i+len(idiom)]

                sim = SequenceMatcher(None, cur, idiom).ratio()

                if sim > 0.9:
                    print('found it!')

                    print('example:', ex)

                    print('idiom:', idiom)

                    print('-----')

                    tag[i:i+len(idiom)] = [1] * len(idiom)

            if ex not in key:
                key[ex] = tag
            else:
                for i, b in enumerate(tag):
                    if b == 1:
                        key[ex][i] = 1

            
    for ex, tag in key.items():

        db['training'].insert_one({'ex': ex, 'tag': tag})

        for c in ex: 
            print('{0:8}'.format(c), end='')

        for t in tag:
            print('{0:8}'.format(t), end='')

        print()
        print()


