import sys
import nltk
import math
import time
import numpy
from collections import defaultdict

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# TODO: IMPLEMENT THIS FUNCTION
# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    brown_words = []
    brown_tags = []
    for sentence in brown_train:
        sentence = sentence.strip()
        tokens = sentence.split(' ')
        words = []
        tags = []
        for token in tokens:
            word,tag = token.rsplit('/',1)
            words.append(word)
            tags.append(tag)
        words.insert(0,START_SYMBOL)
        tags.insert(0,START_SYMBOL)
        words.insert(0,START_SYMBOL)
        tags.insert(0,START_SYMBOL)
        words.append(STOP_SYMBOL)
        tags.append(STOP_SYMBOL)
        brown_words.append(list(words))
        brown_tags.append(list(tags))
    return brown_words, brown_tags


# TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    q_values = {}
        
    bigram_c = defaultdict(float)
    trigram_c = defaultdict(float)
    for tags in brown_tags:
        trigrams = list(nltk.trigrams(tags))
        for trigram in trigrams:
            t1,t2,t3 = trigram
            trigram_c[trigram] = trigram_c[trigram] +1
            bigram_c[tuple([t1,t2])] = bigram_c[tuple([t1,t2])] +1
    for trigram in trigram_c:
        w1,w2,w3 = trigram
        q_values[trigram] = math.log(trigram_c[trigram]/bigram_c[tuple([w1,w2])],2)
        
    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    trigrams.sort()  
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    known_words = set([])
    word_c = defaultdict(int)
    for words in brown_words:
        for word in words:
            word_c[word] = word_c[word]+1
    for word in word_c:
        if word_c[word] > RARE_WORD_MAX_FREQ:
            known_words.add(word)
    return known_words

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = []
    for words in brown_words:
        for i,word in enumerate(words):
            if word not in known_words:
                words[i] = RARE_SYMBOL
        brown_words_rare.append(words)
    return brown_words_rare

# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    e_c = defaultdict(float)
    tags_c = defaultdict(float)
    e_values = {}
    taglist = set([])
    for i in range(len(brown_tags)):
        tags = brown_tags[i]
        words = brown_words_rare[i]
        for j in range(len(tags)):
            e_c[tuple([words[j],tags[j]])] = e_c[tuple([words[j],tags[j]])] + 1
            tags_c[tags[j]] = tags_c[tags[j]] +1
    for e in e_c:
        word,tag = e
        e_values[e] = math.log(e_c[e]/tags_c[tag],2)
    for tag in tags_c:
        taglist.add(tag)
        
    return e_values, taglist

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    emissions.sort()  
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence 
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!
def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    tagged = []
    for words in brown_dev_words:
        viterbi = []
        backpointer = []
        
        v = {}
        b = {}
        v[tuple((START_SYMBOL,START_SYMBOL))] = 0
        b[tuple((START_SYMBOL,START_SYMBOL))] = START_SYMBOL
        viterbi.append(v)
        backpointer.append(b)
        
        for word in words:
            if word not in known_words:
                word = RARE_SYMBOL
            v = {}
            b = {}
            for tag in taglist:
                max = LOG_PROB_OF_ZERO*1000
                for tag1,tag2 in viterbi[-1]:
                    if tuple((word,tag)) in e_values and tuple((tag1,tag2,tag)) in q_values:
                        p = viterbi[-1][tuple((tag1,tag2))]+q_values[tuple((tag1,tag2,tag))]+e_values[tuple((word,tag))]
                        if p > max:
                            max = p
                            v[tuple((tag2,tag))] = p
                            b[tuple((tag2,tag))] = tag1
            if not b:
                for tag in taglist:
                    max = LOG_PROB_OF_ZERO*1000
                    for tag1,tag2 in viterbi[-1]:
                        if tuple((word,tag)) in e_values:                            
                            p = viterbi[-1][tuple((tag1,tag2))]+LOG_PROB_OF_ZERO+e_values[tuple((word,tag))]
                            if p > max:
                                max = p
                                v[tuple((tag2,tag))] = p
                                b[tuple((tag2,tag))] = tag1
                        
            viterbi.append(v)
            backpointer.append(b)
            
        
        max = LOG_PROB_OF_ZERO*1000
        for tag1,tag2 in viterbi[-1]:
            if tuple((tag1,tag2,STOP_SYMBOL)) in q_values:
                p = viterbi[-1][tuple((tag1,tag2))]+q_values[tuple((tag1,tag2,STOP_SYMBOL))]
                if p > max:
                    max = p
                    tags = [tag1,tag2]
        
        for b in reversed(backpointer):
            tag = b[tuple((tags[0],tags[1]))]
            if tag is START_SYMBOL:
                break
            tags.insert(0,tag)
            
        s = ' '.join(['%s/%s' % (words[i],tags[i]) for i in range(len(words))]) + '\n'
            
        tagged.append(s)    
    return tagged

def viterbi_full(brown_dev_words, taglist, known_words, q_values, e_values):
    tagged = []
    for words in brown_dev_words:
        viterbi = []
        backpointer = []
        
        v = {}
        b = {}
        v[tuple((START_SYMBOL,START_SYMBOL))] = 0
        b[tuple((START_SYMBOL,START_SYMBOL))] = START_SYMBOL
        viterbi.append(v)
        backpointer.append(b)
        
        for word in words:
            if word not in known_words:
                word = RARE_SYMBOL
            
            v = {}
            b = {}
            for tag in taglist:
                max = LOG_PROB_OF_ZERO*1000
                for tag1,tag2 in viterbi[-1]:
                    if tuple((word,tag)) in e_values:
                        e = e_values[tuple((word,tag))]
                    else:
                        e = LOG_PROB_OF_ZERO
                    if tuple((tag1,tag2,tag)) in q_values:
                        q = q_values[tuple((tag1,tag2,tag))]
                    else:
                        q = LOG_PROB_OF_ZERO        
                    
                    p = viterbi[-1][tuple((tag1,tag2))]+q
                    if p > max:
                        max = p
                        v[tuple((tag2,tag))] = p+e
                        b[tuple((tag2,tag))] = tag1
                        
            viterbi.append(v)
            backpointer.append(b)
            
        
        max = LOG_PROB_OF_ZERO*1000
        for tag1,tag2 in viterbi[-1]:
            if tuple((tag1,tag2,STOP_SYMBOL)) in q_values:
                q = q_values[tuple((tag1,tag2,STOP_SYMBOL))]
            else:
                q = LOG_PROB_OF_ZERO
            p = viterbi[-1][tuple((tag1,tag2))]+q
            if p > max:
                max = p
                tags = [tag1,tag2]
        
        for b in reversed(backpointer):
            tag = b[tuple((tags[0],tags[1]))]
            if tag is START_SYMBOL:
                break
            tags.insert(0,tag)
            
        s = ' '.join(['%s/%s' % (words[i],tags[i]) for i in range(len(words))]) + '\n'
            
        tagged.append(s)    
    return tagged
    
def viterbi_numpy(brown_dev_words, taglist, known_words, q_values, e_values):
    tagged = []

    N = len(taglist)

    for words in brown_dev_words:
        
        T = len(words)
        
        viterbi = numpy.zeros((T+2,N))
        backpointer = numpy.ones((T+2,N),'int')*list(taglist).index(START_SYMBOL)
        
        v[tuple((START_SYMBOL,START_SYMBOL))] = 0
        b[tuple((START_SYMBOL,START_SYMBOL))] = START_SYMBOL
        viterbi.append(v)
        backpointer.append(b)
        
        for word in words:
            if word not in known_words:
                word = RARE_SYMBOL
            
            v = {}
            b = {}
            for tag in taglist:
                max = LOG_PROB_OF_ZERO*1000
                for tag1,tag2 in viterbi[-1]:
                    if tuple((word,tag)) in e_values:
                        e = e_values[tuple((word,tag))]
                    else:
                        e = LOG_PROB_OF_ZERO
                    if tuple((tag1,tag2,tag)) in q_values:
                        q = q_values[tuple((tag1,tag2,tag))]
                    else:
                        q = LOG_PROB_OF_ZERO        
                    
                    p = viterbi[-1][tuple((tag1,tag2))]+q
                    if p > max:
                        max = p
                        v[tuple((tag2,tag))] = p+e
                        b[tuple((tag2,tag))] = tag1
                        
            viterbi.append(v)
            backpointer.append(b)
            
        
        max = LOG_PROB_OF_ZERO*1000
        for tag1,tag2 in viterbi[-1]:
            if tuple((tag1,tag2,STOP_SYMBOL)) in q_values:
                q = q_values[tuple((tag1,tag2,STOP_SYMBOL))]
            else:
                q = LOG_PROB_OF_ZERO
            p = viterbi[-1][tuple((tag1,tag2))]+q
            if p > max:
                max = p
                tags = [tag1,tag2]
        
        for b in reversed(backpointer):
            tag = b[tuple((tags[0],tags[1]))]
            if tag is START_SYMBOL:
                break
            tags.insert(0,tag)
            
        s = ' '.join(['%s/%s' % (words[i],tags[i]) for i in range(len(words))]) + '\n'
            
        tagged.append(s)    
    return tagged


# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. 
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # following line to format data to what NLTK expects for training
    # brown_words and brown_tags already include START_SYMBOL,START_SYMBOL and STOP_SYMBOL
    training = [ zip(brown_words[i],brown_tags[i]) for i in xrange(len(brown_words)) ]
    
    default_tagger = nltk.DefaultTagger('NOUN')
    unigram_tagger = nltk.UnigramTagger(training, backoff=default_tagger)    
    bigram_tagger = nltk.BigramTagger(training, backoff=unigram_tagger)
    trigram_tagger = nltk.TrigramTagger(training, backoff=bigram_tagger)
    
    tagged = []
    
    for words in brown_dev_words:
        words.insert(0,START_SYMBOL)
        words.insert(0,START_SYMBOL)
        words.append(STOP_SYMBOL)
        tags = trigram_tagger.tag(words)
        s = ' '.join(['%s/%s' % (tags[i][0],tags[i][1]) for i in range(2,len(tags)-1)]) + '\n'
        tagged.append(s)
    
    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare

    # open Brown development data (question 5)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # do viterbi on brown_dev_words (question 5)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 5 output
    q5_output(viterbi_tagged, OUTPUT_PATH + 'B5.txt')

    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 6 output
    q6_output(nltk_tagged, OUTPUT_PATH + 'B6.txt')

    # print total time to run Part B
    print "Part B time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
