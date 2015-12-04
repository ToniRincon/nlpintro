import A
from sklearn.feature_extraction import DictVectorizer
import nltk
from collections import Counter
from sklearn import svm

# You might change the window size
window_size = 15

# B.1.a,b,c,d
def extract_features(data):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :return: features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }
    '''
    features = {}
    labels = {}
     
    for inst in data:
        instance_id = inst[0]
        left_context = inst[1]
        head = inst[2]
        right_context = inst[3]
        sense_id = inst[4]

        f = []
        f = f + nltk.word_tokenize(left_context)[-window_size:]
        f = f + [head]
        f = f + nltk.word_tokenize(right_context)[:window_size]
        
        f = []
        
        try:
            f = f + ['W-2_' + nltk.word_tokenize(left_context)[-2]]
        except:
            pass
        try:
            f = f + ['W-1_' + nltk.word_tokenize(left_context)[-1]]
        except:
            pass
        f + ['W-0_' + head]
        try:
            f = f + ['W+1_' + nltk.word_tokenize(right_context)[0]]
        except:
            pass
        try:
            f = f + ['W+2_' + nltk.word_tokenize(right_context)[1]]
        except:
            pass
        
        fs = Counter(f)
        print fs
        
        features[instance_id] = fs
        labels[instance_id] = sense_id

    return features, labels

# implemented for you
def vectorize(train_features,test_features):
    '''
    convert set of features to vector representation
    :param train_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :param test_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :return: X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
            X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    '''
    X_train = {}
    X_test = {}

    vec = DictVectorizer()
    vec.fit(train_features.values())
    for instance_id in train_features:
        X_train[instance_id] = vec.transform(train_features[instance_id]).toarray()[0]

    for instance_id in test_features:
        X_test[instance_id] = vec.transform(test_features[instance_id]).toarray()[0]

    return X_train, X_test

#B.1.e
def feature_selection(X_train,X_test,y_train):
    '''
    Try to select best features using good feature selection methods (chi-square or PMI)
    or simply you can return train, test if you want to select all features
    :param X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }
    :return:
    '''



    # implement your code here

    #return X_train_new, X_test_new
    # or return all feature (no feature selection):
    return X_train, X_test

# B.2
def classify(X_train, X_test, y_train):
    '''
    Train the best classifier on (X_train, and y_train) then predict X_test labels

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

    :return: results: a list of tuples (instance_id, label) where labels are predicted by the best classifier
    '''

    results = []

    svm_clf = svm.LinearSVC()

    svm_clf.fit([X_train[instance_id] for instance_id in sorted(X_train)],[y_train[instance_id] for instance_id in sorted(X_train)])
    r = svm_clf.predict([X_test[instance_id] for instance_id in sorted(X_test)])
    results = zip(sorted(X_test),r)

    return results

# run part B
def run(train, test, language, answer):
    results = {}

    for lexelt in train:

        train_features, y_train = extract_features(train[lexelt])
        test_features, _ = extract_features(test[lexelt])

        X_train, X_test = vectorize(train_features,test_features)
        X_train_new, X_test_new = feature_selection(X_train, X_test,y_train)
        results[lexelt] = classify(X_train_new, X_test_new,y_train)

    A.print_results(results, answer)