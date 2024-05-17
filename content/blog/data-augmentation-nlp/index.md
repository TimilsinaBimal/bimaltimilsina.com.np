---
title: "Effective Data Augmentation Techniques in NLP"
date: 2023-05-17
tags: ["NLP", "augmentation"]
author: "Bimal Timilsina"
description: "Data augmentation in NLP improves model training by generating new data through methods like back translation, synonym replacement, random swap, random deletion, and sentence paraphrasing. These techniques create diverse, contextually relevant data, enhancing model performance."
categories: ["Natural Language Processing"]
type: post
---

While training any Machine Learning models for large tasks we need similar amount of data. without data the model cannot make the better generalization of the data and it impacts the overall result.

Data Augmentation is a technique which allows us to generate new data after applying some operations in the data we have. while data augmentation is easy and effective in Computer Vision tasks, it can be painful when it comes to NLP. Due to the structure of language, we cannot just replace words in texts. That may lead to completely new sentence. Even if we try to replace words with their synonym, they may result in different sentences altogether. so, it becomes very hard to perform data augmentation methods in NLP.

Here, I will talk about some methods that are proven to be successful in data augmentation in NLP.

### 1. Back Translation

Backtranslation is a data augmentation method in which we translate the original sentence/text to different language and translate it back to the language we need. 

Consider I have a sentence in *English*  , Now using different translate APIs I can translate the English word/sentence into some other language say *Nepali.* Now finally we translate the text in Nepali back to English. While this translation will have same meaning as that of the original sentence, the words used in this sentence will differ by some extent.

Let’s see an example:

**Given the english sentence:**

*Five people have died due to suffocation after a house caught fire in Tulsipur last night.*

**Nepali sentence after translation:**

*तुलसीपुरमा गएराति एक घरमा आगलागी हुँदा निसास्सिएर पाँच जनाको मृत्यु भएको छ*

**Backtranslated sentence from Nepali**

*Five people have died after suffocating in a house fire in Tulsipur last night*

We can see here that while the sentence will have almost same meaning, the words used in the sentence are different. By this we can generate new data in NLP tasks.

### 2. Synonym Replacement

Another method that can be used in data augmentataion method in NLP is synonym replacement. We can select some random words which are not stopwords and replace these words witht their synonyms.

### 3. Random Swap

In this method we randomly choose two words in the sentence and swap their positions. Do this *n* times.

### 4. Random Deletion

In this method we randomly remove each word in the sentence with probability *p*

### 5. Random Replacement

In this method we randomly select some words and add their synonyms in random places.

### 6. Replace Word with Similar word embeddings

Sometimes replacing the words with other words having most similar word embeddings helps in getting better data. Since, word embeddings keep the context in mind, the words generated after replacement will have same meaning.  We can use Word2Vec, GloVe, FastText embeddings to replace words.

### 7. Sentence Paraphrasing

While replacing the words using synonym , and inserting words helps in data augmentation, paraphrasing the sentence will do all the tricks. Paraphrasing simply creates a new sentence which performs operations like synonym replacement, word inserting and so on. We can generate sentences with same meaning but different words using this technique which is the ultimate goal of data augmentation.

Let’s take an example using *parrot* library.

Original Sentence:

*Can you recommed some upscale restaurants in Newyork?*

Paraphrased Sentences:

*recommend some of the best and chic restaurants in new york?
can you suggest some good restaurants in new york city?
which are the best restaurants in newyork?
which are some of the best upscale restaurants in newyork?*

Here, we can see that not only we have replaced some words randomly but also added some context sensitive words to make the sentence more realistic.