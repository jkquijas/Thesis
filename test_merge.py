from nltk import sent_tokenize

data = [['hello','world','how','are','you'],['my','name','is','jonathan']]
labels = [[0,0,0,0,0],[1,1,1,1]]

x = ['my name is earl. This is my story. Thanks.','This is a second sentence. Hi there.']
y = [0,1]

def merge_sentences(s, l, n):
    result = []
    new_labels = []
    n = min(len(s), n)
    temp_data_list = []
    temp_labels_list = []
    for i in range(len(s)-n+1):
        if len(s[i].split()) < 5:
            continue
        temp = ''
        for j in range(n):
            temp += s[i+j] + ' '
        result += [temp]
        new_labels += [l]
    return result, new_labels

def split_into_sentences(data, labels,n=1):
    result = []
    new_labels = []
    for si,s in enumerate(data):
        split_sent, split_labels = merge_sentences(sent_tokenize(s),labels[si],n)
        result += split_sent
        new_labels += split_labels
    return result, new_labels

def test_pass_int(n):
    n = 3
    return n

print split_into_sentences(x,y,1)
n=4
test_pass_int(n)
print n
