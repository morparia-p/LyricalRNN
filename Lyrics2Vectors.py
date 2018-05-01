from gensim.models.fasttext import FastText
from functools import reduce
from collections import Counter
import nltk
import csv
import pickle

def gatherTitlesLyrics(a,b):
  art,title,url,lyr = b
  a['Titles'].append(title)
  a['Lyrics'].append(lyr)
  return a

def loadTitlesLyrics():
  with open("./data/songdata.csv") as lyrics:
    return reduce(gatherTitlesLyrics,csv.reader(lyrics),{"Titles":[],"Lyrics":[]})

def countTokenized(documents):
 return reduce(lambda a,b:a+Counter(b), documents, Counter())

class tokenCounter:
  def __init__(self,tokenizer):
    self.tknzr = tokenizer
  def addCounters(self,a,b):
    return a+Counter(self.tknzr(b))
  def countTokens(self,documents):
    reduce(self.addCounters,documents,Counter())

if __name__ == "__main__":
  d = loadTitlesLyrics()
  d["Lyrics"] = [nltk.tokenize.word_tokenize(l) for l in d["Lyrics"]]
  lyricVectors = FastText(d['Lyrics'],min_count=2,workers=2,size=100)
  lyricVectors.save("LyricVectors.pkl")
  counts = countTokenized(titlesAndLyrics["Lyrics"])
  pickle.dump(counts,"LyricTokenCounts.pkl")
