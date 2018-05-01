from gensim.models.fasttext import FastText
from functools import reduce
import nltk
import csv

def gatherTitlesLyrics(a,b):
  art,title,url,lyr = b
  a['Titles'].append(title)
  a['Lyrics'].append(lyr)
  return a

if __name__ == "__main__":
  with open("./data/songdata.csv") as lyrics:
    d = reduce(gatherTitlesLyrics,csv.reader(lyrics),{"Titles":[],"Lyrics":[]})
  d["Lyrics"] = [nltk.tokenize.word_tokenize(l) for l in d["Lyrics"]]
  lyricVectors = FastText(d['Lyrics'],min_count=2,workers=2,size=100)
  lyricVectors.save("LyricVectors.pkl")
