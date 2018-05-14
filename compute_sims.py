import pandas as pd
import numpy as np

#for name in ['hackathon-train/train/storyzy_en_train.tsv','hackathon-train/train/storyzy_fr_train.tsv','hackathon-train/train/storyzy_yt_train.tsv']:countTrusted(name)

def MWV(text,model):
	text=text.split()
	tmp=np.zeros(300)
	for i in text:
		try:tmp=np.add(tmp,model.wv[i])
		except:pass
	return tmp/len(text)
def ensText(text,model):
	text=text.split()
	tmp=[]
	for i in text:
			try:tmp+=[model.wv[i]]
			except:pass
	return tmp

def meanSim(text,title):
	s=0
	for v in text:s+=sim(v,title)
	return s/len(text)
from gensim.models.wrappers import FastText
#model = FastText.load_fasttext_format('/home/celvaigh/these/divers/wiki.fr/wiki.fr.bin')

fr="wiki.fr.bin"
en="wiki.en.bin"
#model = word_vectors = KeyedVectors.load_word2vec_format('/home/celvaigh/these/divers/wiki.fr/wiki.fr.bin', binary=True)
def computeCorpusSims(name,lg):
	if lg=="fr":model = FastText.load_fasttext_format(fr)
	else:model = FastText.load_fasttext_format(en)
	data=pd.read_csv(name, sep='\t')
	texts=data["text"]
	titles=data["title"]
	size=len(texts)
	sims=[]
	for i in range(size):sims+=[meanSim(ensText(texts[i],model),MWV(titles[i],model))]
	sims.sort()
	return sims

computeCorpusSims('hackathon-train/train/storyzy_fr_train.tsv',"fr")
	
