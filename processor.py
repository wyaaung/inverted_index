import re
import string
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer


class PreProcessor:
    def __init__(self, mwe_tokenizer):
        # Initial Tokenizer to tokenize the raw text.
        self.__tweet_tokenizer = TweetTokenizer()
        # Initializing Translational Table to help remove
        # punctuations and unwanted symbols
        self.__translational_table = str.maketrans(
            "",
            "",
            (string.punctuation).replace("'", "").replace("-", "").replace("#", "")
            + "§―•\t←→",
        )
        # Initializing WordNet Lemmatizer
        self.__wordnet_lemmatizer = WordNetLemmatizer()
        # Stopwords of only english language
        self.__stop_words = set(stopwords.words("english"))
        self.__mwe_tokenizer = mwe_tokenizer

    def pos_tagger(self, nltk_tag):
        """
        Take a POS Tag, and return a wordnet equivalent tag to use for lemmatization
        """
        if nltk_tag == None:
            return None

        if nltk_tag.startswith("J"):
            return wordnet.ADJ
        elif nltk_tag.startswith("V"):
            return wordnet.VERB
        elif nltk_tag.startswith("N"):
            return wordnet.NOUN
        elif nltk_tag.startswith("R"):
            return wordnet.ADV
        else:
            return None

    def get_translational_table(self):
        """
        Return Translational Table helper to remove unwanted symbols
        """
        return self.__translational_table

    def get_wordnet_lemmatizer(self):
        return self.__wordnet_lemmatizer

    def process(self, text_dict: dict) -> dict:
        """
        pre-process a text string and return a list of its terms
        dict->dict
        """

        tokens_dict = {}

        for chapter_name, text_str in text_dict.items():

            # It is a regular expression for finding
            # the pattern for brackets with contents inside them
            raw_text = re.sub("\[.*?\]", "", text_str.lower())

            # Removing punctuations and unnessary symbols
            raw_text_without_puncutations = raw_text.translate(
                self.__translational_table
            )

            # Tokenize the sentences with TweetTokenizer
            tokens_without_punctuations = self.__tweet_tokenizer.tokenize(
                raw_text_without_puncutations
            )

            ############ POS TAGGING AND LEMMATIZATION ON UNIGRAMS ############
            tokens_tags = pos_tag(tokens_without_punctuations)

            # Preparing to lemmatize.
            # Changing from POS Tags to WordNet Tags
            wordnet_tags = [(x[0], self.pos_tagger(x[1])) for x in tokens_tags]

            # Lemmatize with WordNet Lemmatizer
            lemmatized_tokens = [
                word if tag is None else self.__wordnet_lemmatizer.lemmatize(word, tag)
                for word, tag in wordnet_tags
            ]
            ####################################################################

            # Replacing stopwords on Unigrams with placeholder <place_holder>
            lemmatized_uni_tokens_without_sw = [
                word if not word in self.__stop_words else "<place_holder>"
                for word in lemmatized_tokens
            ]

            # Merging multi-word expressions into single tokens
            # using a lexicon of MWEs from CSV Files
            # which creates a new array of tokens
            mixed_tokens = self.__mwe_tokenizer.tokenize(tokens_without_punctuations)
            # Replacing underscore with whitespace to get multi-word tokens
            mixed_tokens = [word.replace("_", " ") for word in mixed_tokens]
            # Removing stopwords on mixed tokens
            multi_tokens_without_sw = [
                word if not word in self.__stop_words else "<place_holder>"
                for word in mixed_tokens
            ]

            multi_tokens_without_sw_uni_index = []
            for token in multi_tokens_without_sw:
                if len(token.split(" ")) == 1:
                    multi_tokens_without_sw_uni_index.append("<place_holder>")
                elif len(token.split(" ")) > 1:
                    multi_tokens_without_sw_uni_index.append(token)
                    for i in range(len(token.split(" ")) - 1):
                        multi_tokens_without_sw_uni_index.append("<place_holder>")

            # Each document will get a list containing two lists of tokens
            # first one unigrams
            # second one multi-grams with indexing intact relative to unigrams
            unigram_multigrams = [
                lemmatized_uni_tokens_without_sw,
                multi_tokens_without_sw_uni_index,
            ]

            tokens_dict[chapter_name] = unigram_multigrams

        return tokens_dict
