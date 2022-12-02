# All Imports
import json
import os
import re
from collections import defaultdict

import pandas as pd
from nltk import pos_tag


class InvertedIndex:
    """
    Inverted Index Class Implementation
    """

    def __init__(self, extractor, preprocessor):
        # For positional indexing, if a term does not exist, it will automatically
        # return "Not In The Index"
        self.__inverted_index = defaultdict(lambda: "Not In The Index")
        self.preprocessor = preprocessor
        self.extractor = extractor

    def size(self):
        """To return the size of inverted index"""
        return len(self.__inverted_index)

    def get_index(self, word: str):
        """Getter method to get inverted index"""
        return self.__inverted_index[word]

    def index_corpus(self, documents: dict) -> None:
        """
        index given documents
        dict->None
        """

        # Iterate through each set of tokens extracted from documents
        for chapter, unigram_multigrams in documents.items():
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

                            # Put chapter if it is not there yet
                            # Then increase the total document frequency
                            if chapter not in self.__inverted_index[term][1]:
                                self.__inverted_index[term][1][chapter] = []
                                self.__inverted_index[term][0] += 1

                            # Initiate the term frequency of specific chapter
                            self.__inverted_index[term][1][chapter].append(1)

                            # Put the positional index of that specific chapter
                            self.__inverted_index[term][1][chapter].append([position])

                        else:
                            # If chapter number in indexing
                            if chapter in self.__inverted_index[term][1]:
                                # if the position of the specific term does not exist yet
                                # in the specific chapter
                                if (
                                    position
                                    not in self.__inverted_index[term][1][chapter][1]
                                ):
                                    # Put the positional index of that specific chapter
                                    self.__inverted_index[term][1][chapter][1].append(
                                        position
                                    )
                                    # Increase the term frequency of specific chapter
                                    self.__inverted_index[term][1][chapter][0] += 1
                            else:
                                # Put chapter number if it is not there yet
                                self.__inverted_index[term][1][chapter] = []
                                # Then increase the total document frequency
                                self.__inverted_index[term][0] += 1
                                # Initiate the term frequency of specific chapter
                                self.__inverted_index[term][1][chapter].append(1)
                                # Put the positional index of that specific chapter
                                self.__inverted_index[term][1][chapter].append(
                                    [position]
                                )

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
                word.lower().translate(self.preprocessor.get_translational_table())
                for word in terms
            ]

            for i in range(len(stripped_terms)):

                # Only lemmatize the unigrams
                if len(stripped_terms[i].split(" ")) < 2:
                    token_tag = pos_tag([stripped_terms[i]])

                    wordnet_tag = (
                        token_tag[0][0],
                        self.preprocessor.pos_tagger(token_tag[0][1]),
                    )

                    lemmatize_token = (
                        self.preprocessor.get_wordnet_lemmatizer().lemmatize(
                            wordnet_tag[0], wordnet_tag[1]
                        )
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
            word.lower().translate(self.preprocessor.get_translational_table())
            for word in terms
        ]

        # Each term will be POS Tagged, Lemmatized
        # then append into processed
        for term in stripped_terms:
            # Only lemmatize the unigrams
            if len(term.split(" ")) < 2:
                token_tag = pos_tag([term])
                wordnet_tag = (
                    token_tag[0][0],
                    self.preprocessor.pos_tagger(token_tag[0][1]),
                )
                lemmatize_token = self.preprocessor.get_wordnet_lemmatizer().lemmatize(
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
