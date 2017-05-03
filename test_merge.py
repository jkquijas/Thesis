data = [['hello','world','how','are','you'],['my','name','is','jonathan']]
labels = [[0,0,0,0,0],[1,1,1,1]]

def merge_sentences(data, labels, n=2):
    result = []
    new_labels = []
    for si,s in enumerate(data):
        n = min(len(s), n)
        l = labels[si][0]
        temp_data_list = []
        temp_labels_list = []
        for i in range(len(s)-n+1):
            temp = ''
            for j in range(n):
                temp += s[i+j] + ' '
            result += [temp]
            new_labels += [l]
        #result += [temp_data_list]
        #new_labels += [temp_labels_list]
    return result, new_labels

print merge_sentences(data,labels,10)
