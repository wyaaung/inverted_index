import time

from inverted_index import InvertedIndex
from extractor import Extractor
from processor import PreProcessor

# Text files' folder path
texts_folder_path = "./Simpsons-2022/"

# Dump text file path
dump_path = "development-examples.txt"


def main():
    "main call function"

    start = time.time()

    extractor = Extractor(texts_folder_path)
    extractor.read_csv()

    preprocessor = PreProcessor(extractor.get_mwe_tokenizer())

    index = InvertedIndex(extractor, preprocessor)  # initilaise the index

    tokens_dict = extractor.read_text()

    corpus = preprocessor.process(tokens_dict)

    index.index_corpus(corpus)

    end = time.time()

    print("EXECUTION TIME: {0:.6f} sec".format(end - start))

    print("Size of Inverted Index : {}\n".format(index.size()))

    return index


index = main()

index.test(dump_path)
