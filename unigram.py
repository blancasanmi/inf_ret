import re
import math
import numpy as np

# used for unseen words in training vocabularies
UNK = None
# sentence start and end
SENTENCE_START = "<s>"
SENTENCE_END = "</s>"


def read_sentences_from_file(file_path):
    '''
        read the files.
    '''
    with open(file_path, "r") as f:
        return [re.split(r"\s+", line.rstrip('\n')) for line in f]


def tf_(doc):  # taken from assignment 4
    frequencies = {}
    for token in doc:
        if token in frequencies:
            frequencies[token] += 1
        else:
            frequencies[token] = 1
    return frequencies


def remove_item(string,
                undesired):  # inspiration from https://stackoverflow.com/questions/1157106/remove-all-occurrences-of-a-value-from-a-list
    return [word for word in string if word != undesired]


class UnigramLanguageModel:
    def __init__(self, sentences, mode="collection", smoothing=True):
        '''
            sentences: sentences of the dataset
            mode: whether this language model is for the whole corpus/collection or just a single document
            smoothing: add-one smoothing
        '''
        # from all the sentences, getting a single list called document with all the relevant words
        self.document = []
        for sent in sentences:
            self.document.extend(sent)

        self.document = remove_item(self.document, SENTENCE_START)
        self.document = remove_item(self.document, SENTENCE_END)

        self.tf = tf_(self.document)
        self.N = len(self.document)
        self.vocab_size = len(self.tf)

        self.smoothing = smoothing
        if smoothing: 
            self.tf[UNK] = 0
            self.vocab_size += 1

    def calculate_unigram_probability(self, word):
        '''
            calculate unigram probability of a word
        '''
        
        if word not in self.tf.keys():
            word = UNK
        word_freq = self.tf[word] 
        
        if not self.smoothing:
            return word_freq / self.N
        return (word_freq + 1) / (self.N + self.vocab_size)  # you need to add 1 to all of them

    def calculate_sentence_probability(self, sentence, normalize_probability=True):
        '''
            calculate score/probability of a sentence or query using the unigram language model.
            sentence: input sentence or query
            normalize_probability: If true then log of probability is not computed. Otherwise take log2 of the probability score.
        '''
        prob = 1
        for word in sentence.split():
            prob *= self.calculate_unigram_probability(word)
            
        if normalize_probability:
            return prob
        else:
            return math.log(prob, 2)


def calculate_interpolated_sentence_probability(sentence, doc, collection, alpha=0.75, normalize_probability=True):
    '''
        calculate interpolated sentence/query probability using both sentence and collection unigram models.
        sentence: input sentence/query
        doc: unigram language model a doc. HINT: this can be an instance of the UnigramLanguageModel class
        collection: unigram language model a collection. HINT: this can be an instance of the UnigramLanguageModel class
        alpha: the hyperparameter to combine the two probability scores coming from the document and collection language models.
        normalize_probability: If true then log of probability is not computed. Otherwise take log2 of the probability score.
    '''
    sentence = remove_item(sentence, SENTENCE_START)
    sentence = remove_item(sentence, SENTENCE_END)

    prob_interpolation = 1
    for word in sentence:
        doc_prob = doc.calculate_unigram_probability(word)
        collec_prob = collection.calculate_unigram_probability(word)
        prob_interpolation *= alpha * doc_prob + (1-alpha) * collec_prob

    if normalize_probability:
        return prob_interpolation
    else:
        return math.log(prob_interpolation, 2)


if __name__ == '__main__':
    # first read the datasets

    actual_dataset = read_sentences_from_file("./train.txt")
    doc1_dataset = read_sentences_from_file("./doc1.txt")
    doc2_dataset = read_sentences_from_file("./doc2.txt")
    doc3_dataset = read_sentences_from_file("./doc3.txt")
    actual_dataset_test = read_sentences_from_file("./test.txt")

    '''
        Question: for each of the test queries given in test.txt, find out best matching document/doc
        according to their interpolated sentence probability.
        Optional: Extend the model to bigram language modeling.
    '''
    # initializing models
    doc1_model = UnigramLanguageModel(doc1_dataset, "document", True)
    doc2_model = UnigramLanguageModel(doc2_dataset, "document", True)
    doc3_model = UnigramLanguageModel(doc3_dataset, "document", True)

    collec_model = UnigramLanguageModel(actual_dataset, "collection", True)

    collection = [doc1_dataset, doc2_dataset, doc3_dataset]
    models = [doc1_model, doc2_model, doc3_model]
    doc_names = ["doc1.txt", "doc2.txt", "doc3.txt"]

    for query in actual_dataset_test:
        print("For query: ", query)
        prob_dict = {}
        for doc_name, doc_model in zip(doc_names, models):
            prob_dict[doc_name] = calculate_interpolated_sentence_probability(query, doc_model, collec_model)
            print(prob_dict[doc_name])
        print("The matched doc is: ", max(prob_dict, key=prob_dict.get))
        
