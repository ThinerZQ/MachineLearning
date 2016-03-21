import bayes

listOPosts ,listClasses = bayes.loadDataSet()

print(listOPosts)

wordVec = bayes.createVocabList(listOPosts)

print(wordVec)

tt = bayes.setOfWords2Vec(wordVec,listOPosts[0])

print(tt)