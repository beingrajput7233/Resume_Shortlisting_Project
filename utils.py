import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize



# Extract text from file, return text
def extract_text(fname):
    myf = open(fname,"rb")
    text = myf.read().decode(errors='replace')
    return text


#Tokenizing the text, return token list
def preprocess(sentence):
 sentence = sentence.lower()
 tokenizer = RegexpTokenizer(r'\w+')
 tokens = tokenizer.tokenize(sentence)
 processed = nltk.word_tokenize(" ".join(tokens))
#  print(processed)
 return  processed


# Stopwords removal, return list
def sw_remove(tokens):
  stop = stopwords.words('english')
  new_tokens = [i for i in tokens if i not in stop]
  return new_tokens


#Stemming of tokens, return list
def stem_tokens(new_tokens):
    ps = PorterStemmer()
    stemmed = []
    for i in new_tokens:
        stemmed.append(ps.stem(i))
    return stemmed

# Helper functions for inv_ind()
class CreateInvDict:
    def __init__(self):
        self.myd = {}

    def checkf(self, x, i):
        if i not in self.myd.keys():
            self.myd[i] = [x]
        else:
            self.myd[i].append(x)


def freq_list(str, word):
    count = str.count(word)
    mid = -1
    freq = []
    for i in range(count):
        prev = str[mid + 1:].index(word)
        mid += (prev + 1)
        freq.append(mid)
    return freq


def freq_count(text, word):
    return text.count(word)


# Inverted index, return dictionary
def inv_ind(stemmed_docs, doc_sizes, n,xfiles):
    unq_tok = set(stemmed_docs)
    inv_table = CreateInvDict()
    for i in unq_tok:
        start = 0
        for j in range(n):
            end = doc_sizes[j]
            temp = stemmed_docs[start:(start + end)]
            if i in temp:
                x = (xfiles[j], freq_count(temp, i))
                inv_table.checkf(x, i)
            start += end
    print("inv table")
    # print(inv_table.myd)
    return inv_table.myd


