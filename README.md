# Inverted Indexing

## Index Term Choices

Many considerations were had when choosing index terms.

The character names and location names in the .csv files were kept as multi-word index terms, as these terms are quite likely candidates for queries.

The set of stop words commonly used for the English language, accessible through nltk.stopwords.words(’english’), was excluded from the list of index terms.

## Preprocessing

Firstly, as much unnecessary phrases in the corpus were deleted before the tokenization phase.

Phrases like ‘From Wikipedia, the free encyclopedia’ and any phrases closed in square brackets were identified by manually going through about 10 articles and making note of phrases that were repeated in all of them without adding anything specific to the article.

Tabs are replaced with whitespaces to counteract some of the weird formatting, and any punctuations and symbols with the exception of apostrophes, sharp symbols, dashes and whitespaces were all removed.

The text is also converted entirely to lowercase. This enables search terms to be case insensitive.

The processed text is split into single-word tokens at this stage through the use of the TweetTokenizer.

After this, a copy of this list of tokens is converted into a mix of multi-word and single-word tokens through the use of MWETokenizer initialized using patterns from the .csv files. Terms produced from MWETokenizer has underscores which are converted to whitespace in these tokens for better presentation.

Compared to lemmatization, stemming increases recall while harming precision. Given that our index is large and likely contains more than enough relevant terms for any query, it seemed wiser to focus on improving precision for potential search queries. Thus, lemmatization was chosen over stemming.

Stop words are converted into placeholder tokens instead of removing them entirely.

Additionally, any single-word tokens inside the list of multi-word tokens are simply ignored upon constructing the inverted index, due to them being duplicate entries with terms from the list of single-word terms.

## What is Stored in the Inverted Index

Each index term included in the inverted index acts as a key, and its corresponding value is a 2-item list.

The first item is the number of documents containing this term, otherwise referred to as ‘document frequency’. The second item is another dictionary that contains file names of the .txt files as keys, and a list as their corresponding values. This list also contains two items; first being the number of occurrence of the index term inside the current file, and second being a list of positions of the index term in the list of pre-processed tokens for this document. This may sound confusing, but essentially the first item of a two-item list stores how many items are stored inside the second item.

The dictionary stored under each index term uses file names as keys instead of a simplified file id.

```
{ “hello” : [ 3 , { ‘4.1' : [ 2 , [ 880 , 882 ] ] , ‘6.18' : [ 1 , [ 1091 ] ] , ‘7.11' : [ 2 , [ 382 , 384 ] ] } ] }
```

For instance, “hello” appears in three chapters whereas of chapter 4.1 at two positions: 881 and 883, as of chapter 6.18 at one position: 1091, and as of chapter 7.11 at two positions: 382 and 384.



## Proximity Search

The proximity search function takes two search terms and a window size, then searches for any occurrences of the two terms in the same document within the specified window size. Any resulting matches are recorded in a dictionary with a similar construction as the inverted index, and is returned.
