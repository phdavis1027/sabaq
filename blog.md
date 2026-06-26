I brushed off some code for an abandoned business idea this week. Abandoned because the business idea was kind of a half-baked excuse for me to work on something that was cool anyway.

Back when I had free time, I loved using it to learn Arabic. I was never much good, but my comprehension was at one point good enough to pick up *some* fairly sophisticated patterns spontaneously. Anyway, Anki was great for drilling vocab. I tried to apply good pedagogical practice by doing a first pass over some text to farm flashcards for unfamiliar words, then practice them, then come back to the text itself once I had the tools to understand it. Problem: Arabic is different enough from English that I often sunk 30 minutes trying to even pin down a word's definition, only to discover that it was being used in an idiom. Idioms are hard to search for even with good reference materials, and Arabic suffers from a startling lack of good references.

So I started playing with [sabaq](https://github.com/phdavis1027/sabaq) ("already" in Arabic). The core product is a tool whose input is some text in your target language, and whose output is that same text with the idioms highlighted. This way, you know which chunks to put in your flashcards.

Step one in reviving this project was to survey the existing code, and God was it bad. The worst code wasn't even AI-generated. I think I was just young, naive, and determined to over-complicate everything. Reading it felt like being forced to listen to a recording of your own voice, so I had Codex get the scraper and the training pipeline back into a runnable and halfway sane-looking state. From there I could iterate.

"What scraper and what training pipeline," you ask? Well, I had decided to start with idiom-detection in French, because I understand it much better (and could therefore judge the results of my algorithm manually) and because the reference materials are way better. Wiktionary basically got me through undergraduate French seminars, so I built a scrapy pipeline that iterated over pages for idioms and captured examples their examples section. This all went got stored in MongoDB, for some stupid reason.

# If you don't know the best solution, pick the funniest one.

Once I had collected every idiom and every example, I used [RapidFuzz](https://rapidfuzz.github.io/RapidFuzz/Usage/distance/LCSseq.html) to find the longest common sub-sequence because each example and its idiom, which turns the examples into labelled training data in a way that is robust against slight variations of the idiom. Consider: "It was, as they say, the best of times, and the worst of times". Naive string-matching would not match this with the idiom "It was the best of times, it was the worst of times."

Now, it occurs to me that this could possibly be sufficient for idiom-detection. Given a bank of idioms large enough to be useful, it *seems* intractable to check for all of them in a non-trivial text. Maybe there are clever ways to reduce this search, but thankfully, I was too stupid to consider this at the time, and instead found a cool paper that solves this problem with an LLM.

Well, by the standards of even open-weight SOTA models today this one is more of an MLM. The paper is called [BERT-based Idiom Identification using Language Translation and Word Cohesion](https://aclanthology.org/2024.mwe-1.26.pdf). It uses [`bert-case-uncased`](https://huggingface.co/google-bert/bert-base-uncased) on Huggingface, which clocks in at a measly 110M params. Compare that with [GLM 5.2](https://huggingface.co/zai-org/GLM-5.2)'s 752B params and you might think this thing is a toy.

One of the many tragedies of the current AI hype cycle is the dogmatic insistence that the only worthwhile research problem is AGI. Besides being delusional, it ignores the fundamental principle of incremental delivery that software engineers were suppose to have learned 30 years ago.

Anyway, the point is that minuscule base models by today's standards can be fine-tuned to solve specific problems with remarkable accuracy. This paper fine-tunes BERT on datasets of no more than about 1,600 examples to label idioms in an input with >= 90% accuracy. And, glory of glories, they posted the code to [Github](https://github.com/phdavis1027/BERT-Based-Idiom-Detection/tree/main). But before I talk about my experience trying to reproduce their results, I simply must tell you about their methods.

As the paper's title suggests, the authors use three loss functions to train their token classification model. The first is a fairly standard binary cross-entropy between the ground-truth labeling and the model's output label. The second loss function is very clever, and the third is completely hilarious.

The clever one is what they call "semantic cohesion." They observe that idioms tend to contain words which are very dissimilar to their surrounding context. For example, a fairly bland tutorial on sysadmin might introduce alternative solutions to a problem by saying, "There's more than one way to skin a cat." In an embedding space, animal mutilation and Bash commands live far away from each other.

To formalize this notion, we can compute cosine similarity between every pair of embeddings in a given sequence of embeddings, forming a matrix that they call a "cohesion graph." To turn this into a loss function, we consider some labeled input. We then compute the difference between the input's cohesion graph and its cohesion graph *after removing the embeddings labelled as idioms*. If the labels are correct, we expect that removing the idiom should increase the sentence's cohesion, so we penalize labelings where this difference is sufficiently small. Incredible!

The hilarious one starts by observing that "an idiom in langauge L1 is unlikely to be an idiomatic phrase in another language L2." Their example is a machine translation into Hindi of the English phrase "raining cats and dogs," which comes out on the other end as भारी वषार्, which is apparently just "heavy rain." So if we take an input, translate it into some very dissimilar language, then translate it back to the input language, we expect idioms to result in a greater difference between the round-tripped translation and the input. Basically, their loss function is that [gag](https://www.youtube.com/watch?v=9DI5WyiHQno) you could do with old Google translate.

They quantify this by using [METEOR](https://aclanthology.org/W05-0909.pdf), which is a metric for automatically evaluating the quality of machine translations by optimizing a mapping between tokens in the original text and the backtranslation respectively. Frankly, I don't entirely understand how METEOR works, but I don't think you lose too much by just thinking of this as saying that we expect idioms to be harder to translate, and therefore we expect that correct labels would have worse METEOR scores. In [code](https://github.com/siddharthyayavaram/BERT-Based-Idiom-Detection/blob/1a14af6e7218b3b172e8e0f85e1b9c59f89b0f2b/src/script.py#L253), this is literally just calls to the Google Translate API per training step.

# Necessity is the mother of optimization

Having snatched their code and swapped out BERT for its Frenchified cousin [CamemBERT](https://huggingface.co/almanach/camembert-base), I set about trying to fine-tune it on my dataset, which was roughly the same size as Yayavaram et al.'s. However, my laptop has 16GB of RAM and on-board AMD graphics. By contrast, to quote Yayavaram et al.: "For training and testing our models, we make use of a 32 × 2 cores AMD EPYC5037532 server with 1 TB of RAM, and 8x A100 SXM4 80GB504." I turned off both cohesion scoring and METEOR scoring, and still my laptop ran at a pace of roughly one training session every 8 hours. Yikes.

After forking out $10 for some Colab time, I ran this baseline almost instantaneously, I think it was like 3 minutes. This resulted in something like 70% recall, roughly matching the paper's report. Then, I turneed on cohesion scoring. This time, it ran for a little bit over an hour and a half but achieved 85% recall. Great!

At this point, it was clear that using the translation metric would take too long to iterate on at all, and I don't have $70 more to spend on fancy Colab GPUs, so I started trying to optimize. Since the only difference between "basically instanteous" and "nearly two hours" was the cohesion computations, I knew that was the place to look.

The authors' subroutine to compute a cohesion graph looks like this:
```
  # Get BERT embeddings for all filtered words in the context
    for word in filtered_context_words:
        word_embeddings[word] = get_bert_embeddings(word, model, tokenizer)

    cohesion_graph = np.zeros((len(filtered_context_words), len(filtered_context_words)))

    # Populate cohesion graph with semantic relatedness scores
    for i, word1 in enumerate(filtered_context_words):
        for j, word2 in enumerate(filtered_context_words):
            cohesion_graph[i, j] = compute_semantic_relatedness(word_embeddings[word1], word_embeddings[word2])
```
That nested loop looks very smelly! When it comes to performance in Python, iteration is root of all evil. We can do better by normalizing the embeddings and stacking them into a matrix. Since each embedding now has length one, pairwise cosine similarity is just their dot product, and so we can compute it for the whole lot by multiplying against the tranpose. When I sanity checked this on a toy example, this is what I got:
```
>>> import numpy as np
>>> from sklearn.metrics.pairwise import cosine_similarity
>>> word_1 = np.array([0.1, 0.9, 2.1])
>>> word_2 = np.array([0.5, 2.5, 0.5])
>>> word_3 = np.array([4.0, 0.0, 0.0])
>>> word_4 = np.array([0.2, 0.2, 0.5])
>>> words = [ word_1, word_2, word_3, word_4 ]
>>> slow = np.zeros((len(words), len(words)))
>>> for i, word1 in enumerate(words):
...     for j, word2 in enumerate(words):
...         slow[i, j] = cosine_similarity([words[i]], [words[j]])[0,0]
...
>>> slow
array([[1.        , 0.56382208, 0.04372695, 0.95148555],
       [0.56382208, 1.        , 0.19245009, 0.56952143],
       [0.04372695, 0.19245009, 1.        , 0.34815531],
       [0.95148555, 0.56952143, 0.34815531, 1.        ]])
>>> stacked = np.stack([word for word in words])
>>> norms = np.linalg.norm(stacked, axis=1, keepdims=True)
>>> norms[norms == 0] = 1.0 
>>> normalized = stacked / norms
>>> normalized @ normalized.T
array([[1.        , 0.56382208, 0.04372695, 0.95148555],
       [0.56382208, 1.        , 0.19245009, 0.56952143],
       [0.04372695, 0.19245009, 1.        , 0.34815531],
       [0.95148555, 0.56952143, 0.34815531, 1.        ]])
```
Neat! Same result, but we went from O(N^2) matrix operations to O(1). Applying this optimization saved me about 20 minutes, for a total runtime of a bit over an hour.

Here's another suspicious loop:
```
def filter_pos(words):
    pos_tags_to_keep = ["NOUN", "VERB"]
    filtered_words = []
    for word in words:
        doc = nlp(word)
        for token in doc:
            if token.pos_ in pos_tags_to_keep:
                filtered_words.append(word)
                break
    return filtered_words
```
Basically, this is applying stopwords to the lemmatized input. Harmless enough on its surface, but `doc = nlp(word)` is a tiny function call that does a lot of damage. It invokes a [SpaCy](spacy.io) pipeline for lemmatization, which is production-grade NLP software itself backed by a beefy neural network. The two problems here are that we run a separate forward pass on every individual word and that we don't cache the tokenizations once we get them. Instead, we could lemmatize the whole input at once and cache results:
```
def _filter_pos(self, words: list[str]) -> list[str]:
	missing = [word for word in dict.fromkeys(words) if word not in self.nlp_cache]
	for word, doc in zip(missing, self.nlp.pipe(missing)):
		self.nlp_cache[word] = any(
			token.pos_ in self.config.pos_tags_to_keep for token in doc
		)
	return [word for word in words if self.nlp_cache[word]]
```
Get this: this change alone brought training time from over an hour to *8.5 minutes*. Now, we could also precompute all of these upfront, but I've been training with 1 epoch, so this was good enough for me to move on to METEOR scoring.


