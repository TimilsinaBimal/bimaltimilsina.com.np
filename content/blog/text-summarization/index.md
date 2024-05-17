---
title: "Enhancing Text Understanding through Summarization Techniques"
date: 2023-05-25
tags: ["NLP", "Summarization"]
author: "Bimal Timilsina"
description: "Text summarization condenses lengthy content through extractive and abstractive methods. Utilizing techniques like TF-IDF, word embeddings, and advanced models such as BERT, it aids in generating concise summaries. Evaluation metrics like ROUGE ensure summary quality, while summarization assists in NLP data augmentation, bridging extractive and abstractive approaches."
categories: ["Natural Language Processing"]
type: post
---

Text summarization is a task where we extract the meaningful information from a long text for concise reading.

When we read a certain research paper, we don’t just go into details of that, we first read the abstract part of it and we find that interersting, then we move forward and research the whole paper. The abstract part gives us clear idea about what the paper is, how the authors have approached the problem and all other methods involved in it. So, this kind of summarization of long text for clear and concise reading is called Text Summarization.

The summary should be related to text and must provide the important points that the paper/article wants to convey. It should also be clear and concise.

## Types

### 1. Extractive Summarization

In extractive text summarization, we try to find the important features from the text and use it to generate the summary. Basically the words in summary will be the words that are present in the text. This is traditional method of text summarization where extract some section of text (paragraph or even words) and merge them to generate the summary.

### 2. Abstractive Summarization

Abstractive summarization is a kind of summarization in which we generate completely new summary from the text based on the main ideas of that text. We can generate abstractive summary using the current SOTA models like BERT.

### Generating Extractive summary from text

To generate the summary, we can use different approaches, like, TF-IDF, Word embeddings, Word Frequency and SOTA models like BERT. These different methods may result in different final output but they all provide the summary of the text.

Here, We will see how we can generate the summary of text using TFIDF Vectorizer method.

### Steps involved in finding the summary of tect using TFIDF Scores

TFIDF (Term Frequency- Inverse document frequency) is a method of text representation. This method gives priority to these words which don’t occur in the text more frequently but has solid impact on the overall meaning of text.

At first we tokenize the texts to find the all the sentences. We can use any NLP library to do the task, but let’s use spacy here:

```python
def preprocess(text):
    nlp = English()
    nlp.add_pipe('sentencizer')
    doc = nlp(text.replace("\n", ""))
    sentences = np.array([sent.text.strip() for sent in doc.sents])
    return sentences
```

Now since we have all the sentences now, we can calculate the tfidf scores of each word in sentences and sum them to get the TFIDF score of a sentence.

```python
def tfidf_vectorizer(sentences):
    tfidf = TfidfVectorizer(
        min_df = 2,
        analyzer='word',
        strip_accents="unicode",
        ngram_range=(1,3),
        stop_words='english'
    )
    return tfidf.fit_transform(sentences)
```

To summarize the text, we need to select top n sentences with highest TFIDF scores:

```python
def top_n_sentences(sentences, tfidf_vectors, n):
    scores = np.sum(tfidf_vectors.toarray(), axis=1).ravel()
    score_idx = np.argsort(scores)[::-1][:n]
    return sentences[score_idx]
```

Now we get the sentences from which we generate the summary. But here the sentences are not in the order they appeared in the text. Since, some texts are in order, we need to take in consideration the order of sentences also. To do that, we create a dictionary using their indices first appeared in text and use them to sort the top sentences.

```python
def sentence_organizer(sentences):
    sent_dict = {
        sent: idx for idx, sent in enumerate(sentences)
    }
    return sent_dict
```

```python
def summarize(text, n_sentences=2):
    sentences = preprocess(text)
    tfidf_vectors = tfidf_vectorizer(sentences)
    top_sentences = top_n_sentences(sentences, tfidf_vectors, n_sentences)
    sentence_dict = sentence_organizer(sentences)

    idx = [sentence_dict[sent] for sent in top_sentences]
    summary = sentences[np.argsort(idx)]
    return "".join(summary)
```

Let’s see an example: <sub class="text-right">Ref:  **The Himalayan Times** [February 14, 2022 ]</sub>

```text
The national active Covid - 19 caseload of Nepal climbed to 22, 584] on Sunday as 427 people tested positive for the infection in past 24 hours.The latest reported number of infections carried the nationwide tally to 973, 059 while the death toll reached 11, 892 as 10 fatalities were recorded today. Meanwhile, the total coronavirus recoveries stand at 938, 583 with 3, 935 discharges logged today.As per the latest data provided by the health ministry, a total of 10, 115 tests were conducted in the last 24 hours of which 6, 037 were PCR tests while 4, 078 were antigen tests. With this, a total of 5, 340, 186 PCR tests have been carried out till date.Similarly, antigen tests have confirmed 94 positive cases in the past 24 hours. The total number of single - day infections from both the RT - PCR and antigen tests totals to 521.Nepal's Covid - 19 recovery rate stands at 96.5 % , while the fatality rate stands at 1.3%.Currently, there are 118 individuals in various quarantine facilities across Nepal.
``` 


Let’s see what our model generates the summary of the above text.

```python
def main():
    text = input("Enter text You want to summarize: ")
    n_sentences = int(input("Enter the number of sentences that you want the text to summarize to: "))
    print(summarize(text,n_sentences))

if __name__ == "__main__":
    main()
```

**OUTPUT**

Top 3 sentences

```text
The national active Covid - 19 caseload of Nepal climbed to 22, 584] on Sunday as 427 people tested positive for the infection in past 24 hours.Meanwhile, the total coronavirus recoveries stand at 938, 583 with 3, 935 discharges logged today.The latest reported number of infections carried the nationwide tally to 973, 059 while the death toll reached 11, 892 as 10 fatalities were recorded today.
```

The summary looks good but certainly not great. It does misses some context and jumps directly to another context. but nonetheless it is a good summary.

## Evaluation Metrics used in Text Summarization

### 1. Human Evaluation

In this method, scores are given by human to each evaluation based on the quality of summarization, grammatical errors, and so on. This is a tedious process, so, we don’t prefer this method.

### 2. Automatic Evaluation

There are not many automatic methods to evaluate the summary of text generated by our model. Here we will discuss about ROUGE score.

### ROUGE

ROUGE stands for *Recall-Oriented Understudy for Gisting Evaluation*. It is the method that determines the quality of the summary by comparing it to other summaries made by humans as a reference. To evaluate the model, there are a number of references created by humans and the generated candidate summary by machine. The intuition behind this is if a model creates a good summary, then it must have common overlapping portions with the human references. It was proposed by [Chin-Yew Lin, University of California](https://www.microsoft.com/en-us/research/publication/rouge-a-package-for-automatic-evaluation-of-summaries/)

There are different versions of ROUGE as given below:

1. **ROUGE-n** 
    
    This is the method of evaluation of summaries of text generated by the automatic model with the summaries generated from humans by comparing the n-grams.
    
    $$
    rouge_n = \frac{p}{q}
    $$
    
    Where,
    
    $p =$ *number of common $n- grams$ in source and reference summaries*
    
    $q =$  *number of n-grams extracted from reference summary*
    

1. **ROUGE-L**
    
    This states that the longest the common subsequence in two texts the common they are. It is a reasonable choice since if the longest common subsequence is all text then both the texts will be similar.
    
2. **ROUGE-SU**
    
    It brings a concept of *skip bi-grams and unigrams.* Basically it allows or considers a bigram if there are some other words between the two words, i.e, the bigrams don’t need to be consecutive words.
    
    ROUGE-2 is most popular and given by:
    
    $$
    Rouge_2 = \frac{\sum_{s\in RefSummaries}{\sum_{bigrams i \in S}min(count(i,X), count(i,S))}}{\sum_{s\in RefSummaries}{\sum_{bigrams i \in S}count(i,S)}}
    $$
    
    Where for every bigram $i$ we calculate the minimum of the number of times it occurred in the generated document $X$ and the reference document $S$, for all reference documents give, divided by the total number of times each bigram appears in all of the reference documents. It is based on BLEU scores.
    

## Extractive Summary in data augmentation

Text summarization can be used in data augmentation of NLP problems. Since the summary of text generates the data that has same meaning as the original text, it can be useful in the NLP data augmentation.

But sometimes the summary generated contains exactly the same words/sentences as that of the original text containing only some valuable information. This kind of summarization is known as extractive summarization. 

Extractive summarization extracts some important sentences from the text and use them to generate the summary of text. Since, it uses all the sentences from the original text, it may not work as a new data in machine learning.

To resolve this we can use data some data augmentation methods in the summarized text like: synonym replacement. This operation helps to generate new words keeping the meaning of text same. This will generate the summary that will be almost same as the summary generated using Abstractive Summarizer.

**Example:** Let’s see an example on how it performs when we use augmentation over extractive summary:

Consider the following piece of text:

```text
The history of natural language processing (NLP) generally started in the 1950s, although work can be found from earlier periods. In 1950, Alan Turing published an article titled "Computing Machinery and Intelligence" which proposed what is now called the Turing test as a criterion of intelligence. The Georgetown experiment in 1954 involved fully automatic translation of more than sixty Russian sentences into English. The authors claimed that within three or five years, machine translation would be a solved problem. However, real progress was much slower, and after the ALPAC report in 1966, which found that ten-year-long research had failed to fulfill the expectations, funding for machine translation was dramatically reduced. Little further research in machine translation was conducted until the late 1980s when the first statistical machine translation systems were developed.
```

Now, To find the extractive summary, I am going to use a library called [Bert Extractive Summarizer](https://github.com/dmmiller612/bert-extractive-summarizer)

```python
from summarizer import Summarizer
model = Summarizer()
model(text)
```

The generated summary is:

```text
The history of natural language processing (NLP) generally started in the 1950s, although work can be found from earlier periods. However, real progress was much slower, and after the ALPAC report in 1966, which found that ten-year-long research had failed to fulfill the expectations, funding for machine translation was dramatically reduced.
```

Since, this contains all the sentences from the original text, let’s replace random words with their synonyms.

```python
def augment(text, method='synonym', use_api=True):
    if method not in ['pegasus', 'parrot', 'synonym']:
        raise ValueError(f"Method {method} not recognized, please use either `pegasus` or `parrot` or `synonym`")
        
    if use_api:
        import requests
        import json
        res = requests.post(
            'https://api.smrzr.io/v1/summarize',
            data=text
        )
        extractive_summary = json.loads(res.text)['summary']
    else:
        model = Summarizer()
        extractive_summary = model(text)
    print(extractive_summary)
    if method == 'pegasus':
        sentences = sent_tokenize(extractive_summary)
        data = [get_response(sent, 1, 10)[0] for sent in sentences]
        print(data)
        return "".join(data)
    elif method == 'synonym':
        augmenter = naw.SynonymAug(stopwords = stopwords.words("english"))
        return augmenter.augment(extractive_summary)
    elif method == 'parrot':
        parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")
        data = [parrot.augment(input_phrase= sent, max_return_phrases=1, use_gpu=False)[0] for sent in sentences]
        return "".join(data)
```

```python
augment(text, method='synonym', use_api=True)
```

On running the program, we get the following output:

**Original Text**

```text
The history of natural language processing (NLP) generally started in the 1950s, although work can be found from earlier periods. In 1950, Alan Turing published an article titled "Computing Machinery and Intelligence" which proposed what is now called the Turing test as a criterion of intelligence. The Georgetown experiment in 1954 involved fully automatic translation of more than sixty Russian sentences into English. The authors claimed that within three or five years, machine translation would be a solved problem. However, real progress was much slower, and after the ALPAC report in 1966, which found that ten-year-long research had failed to fulfill the expectations, funding for machine translation was dramatically reduced. Little further research in machine translation was conducted until the late 1980s when the first statistical machine translation systems were developed.
```

**Original Extractive Summary**

```text
The history of natural language processing (NLP) generally started in the 1950s, although work can be found from earlier periods. However, real progress was much slower, and after the ALPAC report in 1966, which found that ten-year-long research had failed to fulfill the expectations, funding for machine translation was dramatically reduced.
```

**Augmented Extractive Summary**

```text
The history of natural spoken language processing (NLP) generally started in the fifties, although body of work can be witness from earlier periods. Notwithstanding, real progress was a great deal slower, and after the ALPAC report in 1966, which found that tenner - year - long research had failed to fulfill the expectations, fund for machine translation was dramatically reduced.
```

**Original Abstractive Summary**

```text
The history of natural language processing (natural language processing) generally started in the 1950s, although piece of work can be found from early stop. Little further research in automobile translation was conducted until the previous eighties when the first statistical machine translation systems were explicate.
```

Here, we can see that the augmented extractive and abstractive summaries are not exactly similar but they contain almost similar sentences with some modifications in words. This can be considered as a new data while training any models.

We can also paraphrase the extractive summary of text to augment the data. Since paraphrasing also replaces words by their synonyms, insert/remove random words without affecting the meaning of overall text, this also gives a similar result as that of abstractive summary after augmentation.