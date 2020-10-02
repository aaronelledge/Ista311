
'''
Aaron Elledge
05/14/20
ISTA 311
'''


'''
Two implementations of Markov chain text generators.
Character-level generator implemented as a numpy matrix.
Generic token-level generator implemented with dictionaries.
'''

import scipy as sp
from scipy import sparse
from scipy import stats
import string

# Generic character set. Modify this to train the character-level generator with a different character set.
# Restricted character sets may allow training the MC more memory-efficiently.
characters = [ch for ch in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\'\",.?!;:-()[]{} ']

'''
 Token-level text generation. 
 A token is a word or a punctuation mark.
'''

def increment_token(d, token):
    '''
    Auxiliary training function. Increments the count of a certain token in the dictionary by 1.
    '''
    if token in d:
        d[token] += 1
    else:
        d[token] = 1
    return None

'''
def parse_word(word):
    Splits a "word" into a list of tokens. This strips punctuation on the beginning and end of the "word".
    Does not split at internal punctuation; e.g. "o'clock" will not be split into ['o', '\'', 'clock'].
    result = []
    precount = 0
    postcount = 0
    while not( word[0].isalnum() and word[-1].isalnum() ):
        if not word[0].isalnum():
            result.insert(0, word[0])
            word = word[1:]
            precount += 1
        if word == '':
            return result
        if not word[-1].isalnum():
            result.insert(len(result) - postcount, word[-1])
            word = word[:len(word)-1]
            postcount += 1
        if word == '':
            return result
    result.insert(precount, word)
    return result
'''

def create_dist(text, order = 1):
    '''
    Trains the MC model by taking a text, splitting it into a list of tokens, and building a dictionary.
    The keys are pairs whose first elements are tuples of tokens, and second elements are single tokens.
    The values are the counts of how many times the tuple is followed by the associated single token.
    '''
    textlist = []
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    text = ''.join([ch for ch in text if ch not in '0123456789'])
    for word in text.split():
        textlist.append(word)
    newdist = {}
    for i in range(len(textlist)-order):
        increment_token(newdist, (tuple(textlist[i:i+order]), textlist[i+order]))
    return newdist

def add_to_dist(d, text):
    '''
    Retrains an existing MC model by reading a source text and incorporating it into the existing dictionary.
    '''
    #textlist = []
    #for word in text.split():
        #for token in parse_word(word):
    order = len(next(iter(d.keys()))[0])
    textlist = text.split()
    for i in range(len(textlist)-order):
        increment_token(d, (tuple(textlist[i:i+order]), textlist[i+order]))
    return None

def get_next(d, token):
    '''
    Given a trained MC and a token (or tuple of tokens), choose the next token.
    '''
    subset = {k:v for k,v in d.items() if k[0] == token}
    if subset != {}:
        testval = sp.rand() * sum([subset[k] for k in subset])
        for key in subset:
            testval -= subset[key]
            if testval < 0:
                return key[1]
    if token not in d:
        testval = sp.rand() * sum([d[k] for k in d])
        for key in d:
            testval -= d[key]
            if testval < 0:
                return key[1]

def build_string(seq):
    '''
    Condenses a list of tokens into a single string.
    Does not put a space before the tokens listed in stop_punct; does put a space before every other token.
    '''
    string = seq[0]
    stop_punct = ['.', ',', '?', ';', '-', '!']
    for i in range(1, len(seq)):
        if seq[i] in stop_punct:
            string += seq[i]
        else:
            string += (' ' + seq[i])
    return string

def gen_seq(d, length):
    '''
    Generates a sequence of characters from a token dictionary.
    Chooses an initial token (tuple) at random, then calls get_next until the sequence is the desired length.
    '''
    order = len(next(iter(d.keys()))[0])
    seq = []
    testval = sp.rand() * sum([d[k] for k in d])
    for key in d:
        testval -= d[key]
        if testval < 0:
            for s in key[0]:
                seq.append(s)
            break
    while len(seq) <= length:
        token = tuple(seq[len(seq)-order:])
        seq.append(get_next(d, token))
    return build_string(seq)

'''
 Character-level string generation.
 The following functions are for training and simulation of the Markov model at a character level.
 The transition matrix is stored explicitly as a scipy array.
 The array may be sparse or dense.
 All methods default to sparse matrices, since dense matrices are extremely memory-inefficient at any reasonable order.
'''

def lettercode(ch, charset = characters):
    '''
    Returns an encoding of a character, used for indexing the transition matrix.
    '''
    return charset.index(ch)

def idx(token, charset = characters):
    '''
    Computes a row index for a token. Infers order of the MC from the length of the token.
    '''
    m = 0
    n = len(token)
    for i in range(1, n+1):
        m += lettercode(token[-i], charset) * (len(charset) ** (i-1))
    return m

def infer_charset(text):
    return list(set(list(text)))

def create_dense_matrix(order, text, charset = characters):
    '''
    Creates a transition matrix as a dense scipy array.
    '''
    mat = sp.zeros((len(charset) ** order, len(charset)))
    for i in range(len(text)-order-1):
        x_idx = idx(text[i:i+order], charset)
        mat[x_idx, lettercode(text[i+order], charset)] += 1
    return mat

def create_sparse_matrix(order, text, charset = characters):
    '''
    Creates a transition matrix in CSR (compressed sparse row) format.
    '''
    mat = sp.sparse.lil_matrix((len(charset) ** order, len(charset)))
    for i in range(len(text)-order-1):
        x_idx = idx(text[i:i+order])
        mat[x_idx, lettercode(text[i+order])] += 1
    mat = sp.sparse.csr_matrix(mat)
    return mat

def create_matrix(order, text, mode = 'sparse', charset = characters):
    '''
    Reads a text and creates the transition matrix based on the transition frequencies in the text.
    Delegates to sparse- and dense-matrix constructors.
    Dense matrices are somewhat faster, but much more memory-hungry at higher orders.
    '''
    if mode == 'sparse':
        return create_sparse_matrix(order, text, charset)
    if mode == 'dense':
        return create_dense_matrix(order, text, charset)
    else:
        raise NotImplementedError('Unrecognized matrix type. Specify "sparse" or "dense".')

def gen_next(token, mat, charset = characters):
    '''
    Generate the next character in a sequence.
    Slices the transition matrix at the row indicated by the token index, and then chooses a random element.
    '''
    dist = mat[idx(token, charset),:]#.toarray()
    rmax = sum(dist)
    testval = sp.rand() * rmax
    for i in range(len(dist)):
        if testval < dist[i]:
            return charset[i]
        else:
            testval -= dist[i]
    return charset[len(dist)]

def gen_str(n, order, mat, text, charset = characters):
    '''
    Generate a string of length n, using a model of order n given by the transition matrix mat.
    '''
    startpt = sp.random.randint(0, len(text)-order)
    line = text[startpt:startpt+order]
    for i in range(n):
        token = line[len(line)-order:len(line)]
        line = line + gen_next(token, mat, charset)
    return(line)