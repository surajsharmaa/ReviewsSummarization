#import nltk
#nltk.download()

import nltk

#to read excel file
import xlrd

import re

#download all needed data from nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('vader_lexicon')

#read the data
#iphone
fileLoc = "iPhone-XS-Case.xlsx"

#smart watch
#fileLoc="smart-watch.xlsx"

#garage opener
#fileLoc="garage-opener.xlsx"

#Safety glass
#fileLoc="KidsSeftyGlass.xlsx"


wb = xlrd.open_workbook(fileLoc) 
sheet = wb.sheet_by_index(0) 
  
# get total number of rows
row_count = sheet.nrows


#get the frequency distribution
nouns = nltk.FreqDist()        #http://www.nltk.org/_modules/nltk/probability.html 
adjectives = nltk.FreqDist()

#hash of hash with noun being key and all adjectives with their frequencies being value
word_and_adjectives_hash={}

for i in range(row_count):
    #print (i)
    paragrah=sheet.cell_value(i, 0)
    re.sub(' +', ' ', paragrah)
    sentences= nltk.sent_tokenize(paragrah) 
    for data in sentences:
        tokens = nltk.word_tokenize(data) #tokenize the sentences into words
        #remove punctuations
        tokens=[word.lower() for word in tokens if word.isalpha()]
        pos_tags =nltk.pos_tag(tokens)   # POS tag each word using all corpus in NLTK
        #('The', 'AT'), ('grand', 'JJ'), ('jury', 'NN')
        nouns_in_sentence=[]
        adjective_hash=nltk.FreqDist()
        for tagged_word in pos_tags:
            if tagged_word[1]=="NN":
                nouns[tagged_word[0]] +=1
                nouns_in_sentence.append(tagged_word[0])
            if tagged_word[1]=="JJ" or tagged_word[1]=="JJR" or tagged_word[1]=="JJS" :
                #print(tagged_word)
                adjectives[tagged_word[0]] +=1
                adjective_hash[tagged_word[0]] +=1
        
        for eachNoun in nouns_in_sentence:
            for eachAd in adjective_hash:
                if eachNoun in word_and_adjectives_hash:
                    temp = word_and_adjectives_hash[eachNoun]
                    if eachAd in temp:
                        word_and_adjectives_hash[eachNoun][eachAd] += 1
                    else:
                        word_and_adjectives_hash[eachNoun][eachAd] =1
                else:
                    word_and_adjectives_hash[eachNoun]=adjective_hash

#print(nouns.N())
#print(adjectives.N())


#lets lemmatize 

from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

lem_nouns = nltk.FreqDist()        #http://www.nltk.org/_modules/nltk/probability.html 
lem_adjectives = nltk.FreqDist()
for n in nouns:
    lem_nouns[wordnet_lemmatizer.lemmatize(n)] += nouns[n]

for a in adjectives:
    lem_adjectives[wordnet_lemmatizer.lemmatize(a, nltk.corpus.wordnet.ADJ)]+=adjectives[a]   #lemmatized    https://textminingonline.com/dive-into-nltk-part-iv-stemming-and-lemmatization 
    
# print('Lemmatized')
# print(lem_nouns.N())
# print(lem_adjectives.N())


# print("Most common nouns")
# print(lem_nouns.most_common(10))
# print(lem_adjectives.most_common(10))

most_common_nouns=lem_nouns.most_common(10)

# for mcn in most_common_nouns:
#     d = word_and_adjectives_hash[mcn[0]]
#     d= d.most_common(10)
#     print(mcn[0])
#     for i in d:
#         print ("             ",i)



from nltk.corpus import wordnet as wn

word_and_adjectives_hash_synset={}
updated_lem_nouns = nltk.FreqDist()
for mcn in word_and_adjectives_hash:
    #print(mcn)
    #print(word_and_adjectives_hash[mcn])
    for s in wn.synsets(mcn):
        lemmas = s.lemmas()
        for l in lemmas:
            for nn in word_and_adjectives_hash:
                if l.name() == nn:   #match
                    #get len_noun count and add to existing len_noun
                    updated_lem_nouns[nn] += lem_nouns[nn]
                    #print(l.name, nn)
                    if nn in word_and_adjectives_hash_synset:
                        tempad=word_and_adjectives_hash[mcn]
                        temp = word_and_adjectives_hash_synset[nn]
                        for ad in tempad:
                            if ad in temp:
                                word_and_adjectives_hash_synset[nn][ad] += 1
                            else:
                                word_and_adjectives_hash_synset[nn][ad] = 1
                    else:
                        word_and_adjectives_hash_synset[nn]=word_and_adjectives_hash[mcn]
                else:   #doesn't match
                    #print(l.name, nn[0])
                    word_and_adjectives_hash_synset[mcn]=word_and_adjectives_hash[mcn]


#print(len(word_and_adjectives_hash_synset))
#print(len(lem_nouns))

new_most_common_nouns=updated_lem_nouns.most_common(20)
# print("------------------------------------------------------------------------")

for mcn in new_most_common_nouns:
    d = word_and_adjectives_hash_synset[mcn[0]]
    d= d.most_common(10)
    print(mcn[0], mcn[1])
    # for i in d:
    #     print ("             ",i)

#sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

#calculate average score for each feature
for mcn in new_most_common_nouns:
    d = word_and_adjectives_hash_synset[mcn[0]]
    d= d.most_common(100)
    average_score=0
    total_adj=1
    print(mcn[0])
    for i in d:
        cmp_score=sid.polarity_scores(i[0])["compound"]
        #Ignore neutral compound score [0], for negetive score, consider it as 0, and positive as 1
        if cmp_score!=0:
            if cmp_score < 0:
                cmp_score=0
            else:
                cmp_score=1
            ad_score= cmp_score * i[1]
            average_score +=ad_score
            total_adj+=i[1]

    print("              ",(average_score/total_adj)*5)

#create phrases
for mcn in new_most_common_nouns:
    d = word_and_adjectives_hash_synset[mcn[0]]
    d= d.most_common(3)
    generated_sent=""
    print(mcn[0])
    for i in d:
        #print ("              ",i[0],mcn[0])
        generated_sent +=i[0] +", "
    generated_sent += mcn[0]
    print("              ",generated_sent)


print('Done!')