import pickle

def preprocess(lines,word_vocab,char_vocab,max_len):

    word_cnt = len(word_vocab)
    char_cnt = len(char_vocab)+1
    for line in lines:
        words = line.split()+['+']
        for word in words:
            if word not in word_vocab:
                word_vocab[word] = word_cnt
                word_cnt += 1

            for char in word:
                if char not in char_vocab:
                    char_vocab[char] = char_cnt
                    char_cnt += 1

            if max_len < len(word):
                max_len = len(word) 


    return (word_vocab,char_vocab,max_len)
            



f = open("data/train.txt",'r')
train = f.readlines()
f.close()

f = open("data/valid.txt",'r')
valid = f.readlines()
f.close()

f = open("data/test.txt",'r')
test = f.readlines()
f.close()


#make vocab of word and character & get max length of word

word_vocab = {}
char_vocab = {}
max_len = 0

word_vocab,char_vocab,max_len = preprocess(train, word_vocab, char_vocab, max_len)
word_vocab,char_vocab,max_len = preprocess(valid, word_vocab, char_vocab, max_len)
word_vocab,char_vocab,max_len = preprocess(test, word_vocab, char_vocab, max_len)

result = {}

char_vocab['<start>'] = len(char_vocab)+1
char_vocab['<end>'] = len(char_vocab)+1

result['word_vocab'] = word_vocab
result['char_vocab'] = char_vocab
result['max_len'] = max_len

pickle.dump(result,open('data/dic.pickle','wb'))

