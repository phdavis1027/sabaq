# NOTE:
Only the Django portion of this app is under active development. The scraping and learning features are exciting next steps!

# ALSO NOTE:

This project's main host is radicle. Check it out [here](https://app.radicle.xyz/nodes/ash.radicle.garden/rad:z4LYjbg8fB7hzBvQ93K8uZjwycDES)


***Make sure to clone recursive.***

Installing deps (use a venv!):
```bash
pip install -r requirements/dev.txt
pip install requirements/FreNetic
```

Running credit to exciting research that I have referenced for algorithms:

- [BERT-Based Idiom Detection](https://github.com/siddharthyayavaram/BERT-Based-Idiom-Detection/tree/main)
- [Fuzzy dedup using Jaccard similarity](https://blog.nelhage.com/post/fuzzy-dedup/)

We populate the dictionary in two ways:
- Dragnet: hoovering up as many entries as possible and upserting them to
the closest matching entry that we already have
- Refresh: iterating over every entry, maybe checking its added-date, that we already have and pulling up-to-date definitions
from our sources
