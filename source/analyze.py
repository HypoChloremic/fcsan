# Test of fcsreader.py
# (cc) 2017 Ali Rassolie
# Formagna

# An illustration of how access the arguments.
# Den här kommer att spara det the user har entered, vilket är värt att förstå.
# Alltså, när användaren skriver exempelvis python 1.py -a hello world (för att -a
# accepterar två arguemnts) kommer man att spara skiten i en Namespace lista,
# verkar det som, där man har equatat argumenten utan bindestreck med the values 
# man entered som en string!

# We should look on SSC-A on FSC-A, then FSC-H on FSC-W, then
# APC-A on SSC-A, count on APC-A, then APC-A on SSC-A


from  fcsreader import fcsReader
from subprocess import call
from sklearn.cluster import KMeans
from pandas import DataFrame as df
from pandas import concat
from math import log
import matplotlib.pyplot as plt
import seaborn
import os
__version__ = "0.12a"

class Analyze:
	def __init__(self, config="config/config.yaml", file=0, *args, **kwargs):

		self.pos = file

		print("[GLOBAL] Starting the analysis")
		print("[GLOBAL] Opening config")
		with open(config, "r") as file:
			self.config = {i.split(": ")[0]: i.split(": ")[1] for i in file.readlines()}
			self.path   = self.config["PARENT"] 

		print(f"[GLOBAL] The parent-path: {self.path}")
		

	def read(self, file=None, **kwargs):
		self.__files()
		if not file: file=self.file 
		process   = fcsReader(file)
		self.meta = process.meta
		self.dataset = process.data

	def __files(self, top=None, **kwargs):
		if not top: top=self.path
		files = []

		files = [f"{i[0]}\\{k}" for i in os.walk(top) for k in i[-1] if ".fcs" in k]
		self.file = files[self.pos]		

  ########################
  ### Analysis methods ###
  ########################
	def kmeans(self, dataset=None, nclusters=2, logx=False, logy=False, dimensions=2, transpose=False, channels=None, **kwargs):

		print("[ANALYSIS] Running KMeans clustering")
		if dataset is None: dataset = self.dataset

		# Necessary for seaborn, providing the number of 
		# clusters to color
		predict   = KMeans(n_clusters=nclusters, *kwargs).fit(dataset)
		predicted = df(predict.predict(self.dataset), columns=["mapping"])
		dataset = concat([dataset, predicted], axis=1)

		if channels and not transpose: x,y=channels[0],channels[1]
		else: x,y=channels[1],channels[0]
		if logx is True: dataset[x] = dataset[x].apply(func=lambda x: self.log(x))
		if logy is True: dataset[y] = dataset[y].apply(func=lambda y: self.log(y))

		# Using seaborn so as to color map the scatter plot
		order = [i for i in range(nclusters)]
		fg    = seaborn.FacetGrid(data=dataset, hue="mapping", hue_order=order, aspect=1.61)

		fg.map(plt.scatter, "SSC-A", "APC-A").add_legend()
		plt.show()

	def histo(self, dataset=None, channels=None):
		if dataset  is None: dataset=self.dataset
		if channels is None: channels=self.channels

	def log(self, i):
		if i > 0: return log(i)
		else: return None



	def plot(self,dataset=None, xfunc=None, yfunc=None, transpose=False, **kwargs):
		if not dataset: dataset = self.dataset 
		if transpose: 
			p = kwargs["x"]
			kwargs["x"] = kwargs["y"]
			kwargs["y"] = p
		if xfunc: dataset[kwargs["x"]] = dataset[kwargs["x"]].apply(func=lambda x: xfunc(x))
		if yfunc: dataset[kwargs["y"]] = dataset[kwargs["y"]].apply(func=lambda y: yfunc(y))
		self.dataset.plot(**kwargs)
		plt.show()

def _log(i):
	if i > 0: return log(i)
	else: return None


if __name__ == '__main__':
	run = Analyze()
	run.read()
	run.kmeans(channels=["SSC-A", "APC-A"], logx=True, logy=True, transpose=True)
	run.plot(x="APC-A", y="SSC-A", xfunc=_log, yfunc=_log, kind="scatter", transpose=True)