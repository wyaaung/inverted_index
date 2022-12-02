import os
import pandas as pd

from nltk.tokenize import MWETokenizer


class Extractor:
    def __init__(self, filepath: str):
        # MWETokenizer will be initialized later after reading CSV Files
        self.__mwe_tokenizer = MWETokenizer()
        # Used Tenary Operator to add slash
        # if filepath does not have a slash at the end
        self.filepath = filepath if filepath[-1] == "/" else filepath + "/"
        self.files = os.listdir(filepath)
        self.csv_read = False

    def get_mwe_tokenizer(self):
        """
        Return Loaded MWETokenizer
        """
        return (
            self.__mwe_tokenizer
            if self.csv_read
            else Exception("You have to read CSV Tokens First")
        )

    def read_csv(self):
        """
        Read all the csv files under the filepath,
        creating a list of proper nouns,
        using to initialise Multi-word tokenizer
        """

        # to hold all tokens from CSV Files
        special_tokens = []

        # Look through all the filename from above list,
        # extracting .txt files
        csv_files = [filename for filename in self.files if ".csv" in filename]

        # Iterate all csv files
        for filename in csv_files:
            names = pd.read_csv(self.filepath + filename)

            # Turning the column "name" of the dataframe into the list
            names = names["name"].to_list()

            # Appended all the names from columns of dataframe into special_tokens
            special_tokens += names

        # Add Multiword Expressions into MWETokenizer
        # Will be used later in preprocessing
        [
            self.__mwe_tokenizer.add_mwe(tuple(word.lower().split(" ")))
            for word in special_tokens
        ]

        self.csv_read = True

    def read_text(self) -> dict:
        """
        Read files from a directory and then append the data of each file into a list.
        """

        # # Placeholder to storce lists of tokens extracted from the txt files
        tokens_dict = {}

        # Look through all the filename from above list,
        # extracting .txt files
        text_files = sorted([filename for filename in self.files if ".txt" in filename])

        for filename in text_files:
            with open(
                self.filepath + "/" + filename, "r", encoding="utf-8-sig"
            ) as file:
                raw_text = file.read()

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

            chapter_name = filename.rsplit(".", 1)[0]

            if chapter_name not in tokens_dict:
                tokens_dict[chapter_name] = ""

            tokens_dict[chapter_name] += raw_text

        return tokens_dict
