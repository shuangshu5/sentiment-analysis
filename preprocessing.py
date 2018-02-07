#-*- coding: UTF-8 -*-
import pandas as pd
import jieba
import re
import sys
reload(sys)
sys.setdefaultencoding('utf8')

# Load data
comments = []
labels = []
data = open('data/comments.txt', 'r')
for line in data.readlines():
    try:
        line = line.strip()
        comment = unicode(line[:-1])
        comment = re.sub(u"[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。“”’‘？?、~@#￥%……&*（）]+", "", comment)
        label = int(line[-1])
        comments.append(comment)
        labels.append(label)
    except:
        pass
data.close()

# Load stopwords
stopwords = []
s = open("data/stop_words.txt", 'r')
for word in s.readlines():
    stopwords.append(word.strip().decode('gbk'))

# Preprocessing
num = 0
contents = []
for comment in comments:
    words = jieba.lcut(comment)
    for word in words:
        if word in stopwords:
            words.remove(word)
    contents.append(words)
    num += 1
    print u'总共有%d条数据,已处理成功%d条数据' % (len(comments),num)

postive = open('data/positive.txt', "w")
negtive = open('data/negtive.txt', "w")
p = 0
n = 0
for i in range(len(labels)):
    if labels[i] > 3:
        p += 1
        for word in contents[i]:
            postive.write(word.encode('utf-8')+" ")
        postive.write("\n")

    else:
        n += 1
        for word in contents[i]:
            negtive.write(word.encode('utf-8')+" ")
        negtive.write("\n")

print 'postive: %d' % p, 'negtive: %d' % n

postive.close()
negtive.close()