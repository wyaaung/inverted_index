# All Imports
import json
import os
import re
import string
from collections import defaultdict

import pandas as pd
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import MWETokenizer, TweetTokenizer


class InvertedIndex:
    """
    Inverted Index Class Implementation
    """

    def __init__(self):
        # Initial Tokenizer to tokenize the raw text.
        self.__tweet_tokenizer = TweetTokenizer()
        # MWETokenizer will be initialized later after reading CSV Files
        self.__mwe_tokenizer = MWETokenizer()
        # Stopwords of only english language
        self.__stop_words = set(stopwords.words("english"))
        # Used defaultdict to store the file number and Chapter Numbers
        self.__chapter_map = defaultdict(lambda: "No Such Chapter Existed")
        # For positional indexing, if a term does not exist, it will automatically
        # return "Not In The Index"
        self.__inverted_index = defaultdict(lambda: "Not In The Index")
        # Initializing WordNet Lemmatizer
        self.__wordnet_lemmatizer = WordNetLemmatizer()
        # Initializing Translational Table to help remove
        # punctuations and unwanted symbols
        self.__translational_table = str.maketrans(
            "",
            "",
            (string.punctuation).replace("'", "").replace("-", "").replace("#", "")
            + "§―•\t←→",
        )

    def size(self):
        """To return the size of inverted index"""
        return len(self.__inverted_index)

    def get_inverted_index(self):
        """Getter method to get inverted index"""
        return self.__inverted_index

    def __pos_tagger(self, nltk_tag):
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

    def __read_csv(self, path: str, csv_files: list) -> list:
        """
        Read all the csv files under the path,
        creating a list of proper nouns,
        using to initialise Multi-word tokenizer
        """

        # to hold all tokens from CSV Files
        special_tokens = []

        # Iterate all csv files
        for filename in csv_files:
            names = pd.read_csv(path + filename)

            # Turning the column "name" of the dataframe into the list
            names = names["name"].to_list()

            # Appended all the names from columns of dataframe into special_tokens
            special_tokens += names

        [
            self.__mwe_tokenizer.add_mwe(tuple(word.lower().split(" ")))
            for word in special_tokens
        ]

    def read_data(self, path: str) -> list:
        """
        Read files from a directory and then append the data of each file into a list.
        """

        # Placeholder to storce lists of tokens extracted from the txt files
        tokens_list = []

        # Acquring all the file names inside that path
        files = os.listdir(path)

        # Look through all the filename from above list,
        # extracting .txt files and .csv files
        text_files = sorted([filename for filename in files if ".txt" in filename])
        csv_files = [filename for filename in files if ".csv" in filename]

        # add slash if path does not have a slash at the end
        if path[-1] != "/":
            path = path + "/"

        # Passing csv files list and path into "Private" reading CSV method
        self.__read_csv(path, csv_files)

        # File number start from zero
        chapter_number = 0

        # Iternate through all files
        # Extract the content of the file as string,
        # Preprocess each files
        # Take pre-processed tokens and append into placeholder I declared above
        for filename in text_files:
            with open(path + "/" + filename, "r", encoding="utf-8-sig") as file:
                raw_text = file.read()
            file.close()

            # Remove texts appear in every file
            raw_text = raw_text.replace("From Wikipedia, the free encyclopedia", "")
            raw_text = raw_text.replace("Jump to navigation", "")
            raw_text = raw_text.replace("Jump to search", "")
            raw_text = raw_text.replace("← Previous", "")
            raw_text = raw_text.replace("Next →", "")
            raw_text = raw_text.replace("Plot", "")
            raw_text = raw_text.replace("Production", "")
            raw_text = raw_text.replace("Reception", "")
            raw_text = raw_text.replace("References", "")
            raw_text = raw_text.replace("External links", "")
            raw_text = raw_text.replace("The Simpsons episode", "")

            # Map chapter_number and chapter number
            self.__chapter_map[chapter_number] = filename.rsplit(".", 1)[0]

            tokens = self.process_document(raw_text)

            chapter_number += 1

            tokens_list.append(tokens)

        return tokens_list

    def process_document(self, document: str) -> list:
        """
        pre-process a document and return a list of its terms
        str->list"""
        # It is a regular expression for finding
        # the pattern for brackets with contents inside them
        raw_text = re.sub("\[.*?\]", "", document.lower())

        # Removing punctuations and unnessary symbols
        raw_text_without_puncutations = raw_text.translate(self.__translational_table)

        # Tokenize the sentences with TweetTokenizer
        tokens_without_punctuations = self.__tweet_tokenizer.tokenize(
            raw_text_without_puncutations
        )

        ############ POS TAGGING AND LEMMATIZATION ON UNIGRAMS ############
        tokens_tags = pos_tag(tokens_without_punctuations)

        # Preparing to lemmatize.
        # Changing from POS Tags to WordNet Tags
        wordnet_tags = [(x[0], self.__pos_tagger(x[1])) for x in tokens_tags]

        # Lemmatize with WordNet Lemmatizer
        lemmatized_tokens = [
            word if tag is None else self.__wordnet_lemmatizer.lemmatize(word, tag)
            for word, tag in wordnet_tags
        ]
        ####################

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

        return unigram_multigrams

    def index_corpus(self, documents: list) -> None:
        """
        index given documents
        list->None"""

        # Iterate through each set of tokens extracted from documents
        for chapter_number, unigram_multigrams in enumerate(documents):
            for tokens in unigram_multigrams:
                for position, term in enumerate(tokens):
                    # Skip place_holder tokens
                    if term != "<place_holder>":
                        if term not in self.__inverted_index:
                            # Initialize the list
                            self.__inverted_index[term] = []
                            # The total document frequency is initialize as 0
                            self.__inverted_index[term].append(0)
                            # Initialize dictionary to store chapter number
                            self.__inverted_index[term].append({})

                            # Put chapter number if it is not there yet
                            # Then increase the total document frequency
                            if (
                                self.__chapter_map[chapter_number]
                                not in self.__inverted_index[term][1]
                            ):
                                self.__inverted_index[term][1][
                                    self.__chapter_map[chapter_number]
                                ] = []
                                self.__inverted_index[term][0] += 1

                            # Initiate the term frequency of specific chapter number
                            self.__inverted_index[term][1][
                                self.__chapter_map[chapter_number]
                            ].append(1)

                            # Put the positional index of that specific chapter number
                            self.__inverted_index[term][1][
                                self.__chapter_map[chapter_number]
                            ].append([position])

                        else:
                            # If chapter number in indexing
                            if (
                                self.__chapter_map[chapter_number]
                                in self.__inverted_index[term][1]
                            ):
                                # if the position of the specific term does not exist yet
                                # in the specific chapter number
                                if (
                                    position
                                    not in self.__inverted_index[term][1][
                                        self.__chapter_map[chapter_number]
                                    ][1]
                                ):
                                    # Put the positional index of that specific chapter number
                                    self.__inverted_index[term][1][
                                        self.__chapter_map[chapter_number]
                                    ][1].append(position)
                                    # Increase the term frequency of specific chapter number
                                    self.__inverted_index[term][1][
                                        self.__chapter_map[chapter_number]
                                    ][0] += 1
                            else:
                                # Put chapter number if it is not there yet
                                self.__inverted_index[term][1][
                                    self.__chapter_map[chapter_number]
                                ] = []
                                # Then increase the total document frequency
                                self.__inverted_index[term][0] += 1
                                # Initiate the term frequency of specific chapter number
                                self.__inverted_index[term][1][
                                    self.__chapter_map[chapter_number]
                                ].append(1)
                                # Put the positional index of that specific chapter number
                                self.__inverted_index[term][1][
                                    self.__chapter_map[chapter_number]
                                ].append([position])

        # Save inverted index with JSON Format File
        self.__dump()

    def __dump(self) -> None:
        """
        Dump an inverted index as of JSON Format File.
        """
        with open("inverted_index.json", "w", encoding="utf-8-sig") as file:
            json.dump(self.__inverted_index, file)

    def test(self, path: str) -> None:
        """
        provide a test function to show index entries for a given set of terms
        """

        # Open and extract the content of the file as string,
        try:
            with open(path, "r", encoding="utf-8-sig") as file:
                _text = file.read()

            file.close()

            # Split according to new line character
            terms = _text.split("\n")

            # Remove punctuations and unwanted symbols
            stripped_terms = [
                word.lower().translate(self.__translational_table) for word in terms
            ]

            for i in range(len(stripped_terms)):

                # Only lemmatize the unigrams
                if len(stripped_terms[i].split(" ")) < 2:
                    token_tag = pos_tag([stripped_terms[i]])

                    wordnet_tag = (token_tag[0][0], self.__pos_tagger(token_tag[0][1]))

                    lemmatize_token = self.__wordnet_lemmatizer.lemmatize(
                        wordnet_tag[0], wordnet_tag[1]
                    )

                    index = self.__inverted_index[lemmatize_token]

                    print(
                        "Input Term: "
                        + terms[i]
                        + ", POS Tag: "
                        + token_tag[0][1]
                        + ", Lemmatized Term: "
                        + lemmatize_token
                    )

                else:
                    index = self.__inverted_index[stripped_terms[i]]

                    print("Input Term: " + terms[i])

                if index != "Not In The Index":
                    pos_index = index[1]
                    total_term_frequency = 0
                    for chapter_number, positional_indexes in pos_index.items():
                        print(
                            "Chapter: "
                            + chapter_number
                            + ", Term Frequency: {}, ".format(positional_indexes[0])
                            + "Positional Indices: "
                            + ",".join(str(index) for index in positional_indexes[1])
                        )
                        total_term_frequency += positional_indexes[0]

                    print(
                        "Total Term Frequency Across Documents: {}".format(
                            total_term_frequency
                        )
                    )
                    print("Total Document Frequency: {}".format(index[0]))

                else:
                    print(index)
                print()
        except:
            print("There is no such filename.")

    def __process_terms(self, terms: list) -> list:
        """
        Helper function to process terms and return them back
        """
        # Placeholder for processed terms. Will be returned
        processed = []

        # Remove punctuations from tokens
        stripped_terms = [
            word.lower().translate(self.__translational_table) for word in terms
        ]

        # Each term will be POS Tagged, Lemmatized
        # then append into processed
        for term in stripped_terms:
            # Only lemmatize the unigrams
            if len(term.split(" ")) < 2:
                token_tag = pos_tag([term])
                wordnet_tag = (token_tag[0][0], self.__pos_tagger(token_tag[0][1]))
                lemmatize_token = self.__wordnet_lemmatizer.lemmatize(
                    wordnet_tag[0], wordnet_tag[1]
                )

                processed.append(lemmatize_token)
            else:
                processed.append(term)

        return processed

    def proximity_search(self, term1: str, term2: str, window_size: int) -> dict:
        """
        This function takes two search terms and a window size,
        then searches for any occurrences of the two terms
        in the same document within the specified window size
        """

        # Placeholder for chapter number and positional terms
        # which both terms exist
        answers = {}

        # Processing both terms - look at the function implementation above
        processed_terms = self.__process_terms([term1, term2])

        # Acquring positional indices of all chapters which terms are included
        term1_chapter_indices = self.__inverted_index[processed_terms[0]]
        term2_chapter_indices = self.__inverted_index[processed_terms[1]]

        if (
            term1_chapter_indices == "Not In The Index"
            or term2_chapter_indices == "Not In The Index"
        ):
            print("One of each terms does not exist in the inverted index.")
            return answers

        # Acquiring common chapters which both terms are there
        common_chapters = [
            key for key in term1_chapter_indices[1] if key in term2_chapter_indices[1]
        ]

        for chapter in common_chapters:
            term1_indexes = term1_chapter_indices[1][chapter]
            term2_indexes = term2_chapter_indices[1][chapter]

            for term1_index in term1_indexes[1]:
                for term2_index in term2_indexes[1]:
                    # Add into answer dict if window size is greater than or equal to
                    # the absolute difference of position of second term and that of first term
                    if abs(term2_index - term1_index) < window_size:
                        if chapter not in answers:
                            answers[chapter] = [1, [(term1_index, term2_index)]]
                        else:
                            answers[chapter][0] += 1
                            answers[chapter][1].append((term1_index, term2_index))
        return answers
