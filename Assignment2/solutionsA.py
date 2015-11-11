import math
import nltk
import time
from collections import defaultdict

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    unigram_c = defaultdict(float)
    bigram_c = defaultdict(float)
    trigram_c = defaultdict(float)
    for sentence in training_corpus:
        sentence = sentence.strip()
        tokens = sentence.split(' ')
        tokens.append(STOP_SYMBOL)
        tokens.insert(0,START_SYMBOL)
        for unigram in tokens:
            unigram_c[tuple([unigram])] = unigram_c[tuple([unigram])]+1
        tokens.insert(0,START_SYMBOL)        
        for bigram in list(nltk.bigrams(tokens)):
            bigram_c[bigram] = bigram_c[bigram] +1        
        for trigram in list(nltk.trigrams(tokens)):
            trigram_c[trigram] = trigram_c[trigram] +1
    unigram_p = {}
    bigram_p = {}
    trigram_p = {}
    for trigram in trigram_c:
        w1,w2,w3 = trigram
        trigram_p[trigram] = math.log(trigram_c[trigram]/bigram_c[tuple([w1,w2])],2)
    for bigram in bigram_c:
        w1,w2 = bigram
        bigram_p[bigram] = math.log(bigram_c[bigram]/unigram_c[tuple([w1])],2)
    del unigram_c[tuple([START_SYMBOL])]
    total = sum([unigram_c[unigram] for unigram in unigram_c])
    for unigram in unigram_c:
        w1 = unigram
        unigram_p[unigram] = math.log(unigram_c[unigram]/total,2)
    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()    
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, corpus):
    scores = []
    for sentence in corpus:
        score = 0
        sentence = sentence.strip()
        tokens = sentence.split(' ')
        if n == 2:
            tokens.insert(0,START_SYMBOL)
            tokens.insert(0,START_SYMBOL)
            tokens.append(STOP_SYMBOL)
            tokens = list(nltk.bigrams(tokens))
        elif n == 3:
            tokens.insert(0,START_SYMBOL)
            tokens.insert(0,START_SYMBOL)
            tokens.append(STOP_SYMBOL)
            tokens = list(nltk.trigrams(tokens))
        else:
            tokens.insert(-1,STOP_SYMBOL)            
            tokens = list([tuple([token]) for token in tokens])
        for token in tokens:
            if token not in ngram_p:
                score = MINUS_INFINITY_SENTENCE_LOG_PROB
                break
            score = score + ngram_p[token]
        scores.append(score)
    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    
    linear = {}
    scores = []    
    for trigram in trigrams:
        w1,w2,w3 = trigram
        linear[trigram] = math.log(1./3*2**trigrams[tuple([w1,w2,w3])]+1./3*2**bigrams[tuple([w2,w3])]+1./3*2**unigrams[tuple([w3])],2)
        # print trigram,trigrams[tuple([w1,w2,w3])],bigrams[tuple([w2,w3])],unigrams[tuple([w3])],linear[trigram]
    for sentence in corpus:
        sentence = sentence.strip()
        tokens = sentence.split(' ')
        tokens.insert(0,START_SYMBOL)
        tokens.insert(0,START_SYMBOL)
        tokens.append(STOP_SYMBOL)
        score = 0
        for token in list(nltk.trigrams(tokens)):            
            if token not in linear:
                score = MINUS_INFINITY_SENTENCE_LOG_PROB
                break
            score = score + linear[token]
        scores.append(score)
    return scores

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close() 

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
