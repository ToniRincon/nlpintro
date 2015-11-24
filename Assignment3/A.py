from main import replace_accented
from sklearn import svm
from sklearn import neighbors
from collections import Counter
import nltk
import codecs

# don't change the window size
window_size = 10

# A.1
def build_s(data):
    '''
    Compute the context vector for each lexelt
    :param data: dic with the following structure:
        {
			lexelt: [(instance_id, left_context, head, right_context, sense_id), ...],
			...
        }
    :return: dic s with the following structure:
        {
			lexelt: [w1,w2,w3, ...],
			...
        }

    '''
    s = {}


    for lexelt in data:
        inst_list = data[lexelt]
        ws = set()
        for inst in inst_list:
            instance_id = inst[0]
            left_context = inst[1]
            head = inst[2]
            right_context = inst[3]
            sense_id = inst[4]

            ws.update(nltk.word_tokenize(left_context)[-10:])
            ws.update([head])
            ws.update(nltk.word_tokenize(right_context)[:10])
        
        s[lexelt] = list(ws)

    return s


# A.1
def vectorize(data, s):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :param s: list of words (features) for a given lexelt: [w1,w2,w3, ...]
    :return: vectors: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }

    '''
    vectors = {}
    labels = {}

    for inst in data:
        instance_id = inst[0]
        left_context = inst[1]
        head = inst[2]
        right_context = inst[3]
        sense_id = inst[4]

        l = []
        l = l + nltk.word_tokenize(left_context)[-10:]
        l = l + nltk.word_tokenize(right_context)[:10]
        l = l+ [head]
        c = Counter(l)
        ws = [c[w] for w in s]
        vectors[instance_id] = ws
        labels[instance_id] = sense_id

    return vectors, labels


# A.2
def classify(X_train, X_test, y_train):
    '''
    Train two classifiers on (X_train, and y_train) then predict X_test labels

    :param X_train: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param X_test: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }

    :return: svm_results: a list of tuples (instance_id, label) where labels are predicted by LinearSVC
             knn_results: a list of tuples (instance_id, label) where labels are predicted by KNeighborsClassifier
    '''

    svm_results = []
    knn_results = []

    svm_clf = svm.LinearSVC()
    knn_clf = neighbors.KNeighborsClassifier(5,weights='uniform')

    svm_clf.fit([X_train[instance_id] for instance_id in sorted(X_train)],[y_train[instance_id] for instance_id in sorted(X_train)])
    r = svm_clf.predict([X_test[instance_id] for instance_id in sorted(X_test)])
    svm_results = zip(sorted(X_test),r)

    knn_clf.fit([X_train[instance_id] for instance_id in sorted(X_train)],[y_train[instance_id] for instance_id in sorted(X_train)])
    r = knn_clf.predict([X_test[instance_id] for instance_id in sorted(X_test)])
    knn_results = zip(sorted(X_test),r)

    return svm_results, knn_results

# A.3, A.4 output
def print_results(results ,output_file):
    '''

    :param results: A dictionary with key = lexelt and value = a list of tuples (instance_id, label)
    :param output_file: file to write output

    '''

    # implement your code here
    # don't forget to remove the accent of characters using main.replace_accented(input_str)
    # you should sort results on instance_id before printing

    
    f = codecs.open(output_file, encoding='utf-8', mode='w')
    for lexelt in results:
        for instance_id,label in results[lexelt]:
            f.write(replace_accented(lexelt + ' ' + instance_id + ' ' + label + '\n'))
    f.close()

# run part A
def run(train, test, language, knn_file, svm_file):
    s = build_s(train)
    svm_results = {}
    knn_results = {}
    for lexelt in s:
        X_train, y_train = vectorize(train[lexelt], s[lexelt])
        X_test, _ = vectorize(test[lexelt], s[lexelt])
        svm_results[lexelt], knn_results[lexelt] = classify(X_train, X_test, y_train)

    print_results(svm_results, svm_file)
    print_results(knn_results, knn_file)



