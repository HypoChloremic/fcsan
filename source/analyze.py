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
from pandas import concat, cut
from math import log
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import use
import seaborn
import os
__version__ = "0.12a"

class Analyze:
	def __init__(self, config="config/config.yaml", pos=0, name=None, *args, **kwargs):

		self.pos  = pos
		self.name = name

		print("[GLOBAL] Starting the analysis")
		print("[GLOBAL] Opening config")
		with open(config, "r") as file:
			self.config = {i.split(": ")[0]: i.split(": ")[1].replace("\n", "") for i in file.readlines()}
			print(self.config)
			self.path   = self.config["PARENT"] 
			print(self.path)

		print(f"[GLOBAL] The parent-path: {self.path}")
		

	def read(self, file=None, **kwargs):
		if not file: 
			self.__files()
			file=self.file 
		process   = fcsReader(file)
		self.meta = process.meta
		self.dataset = process.data

	def __files(self, top=None, delimiter="\\", **kwargs):
		if not top: top=self.path
		print(top)
		self.files = [f"{i[0]}{delimiter}{k}" for i in os.walk(top) for k in i[-1] if k.endswith(".fcs")]
		if self.name: self.file = [i for i in self.files if self.name in i][0]
		else: 
			self.file = self.files[self.pos]		
			self.names = [f"{i.split(delimiter)[-2]}_{i.split(delimiter)[-1]}" for i in self.files]

  ########################
  ### Analysis methods ###
  ########################
	def kmeans(self, dataset=None, nclusters=2, logx=False, logy=False, dimensions=2, transpose=False, channels=None, **kwargs):

		print("[KMeans] Running KMeans clustering")
		if dataset is None: dataset = self.dataset

		# Necessary for seaborn, providing the number of 
		# clusters to color
		predict   = KMeans(n_clusters=nclusters, **kwargs).fit(dataset)
		predicted = df(predict.predict(self.dataset), columns=["mapping"])
		dataset = concat([dataset, predicted], axis=1)

		if channels and not transpose: x,y=channels[0],channels[1]
		elif channels and transpose:   x,y=channels[1],channels[0]

		print(f"[KMeans] x: {x}, y: {y}")

		if logx is True: dataset[x] = dataset[x].apply(func=lambda x: self.log(x))
		if logy is True: dataset[y] = dataset[y].apply(func=lambda y: self.log(y))

		# Using seaborn so as to color map the scatter plot
		order = [i for i in range(nclusters)]
		fg    = seaborn.FacetGrid(data=dataset, hue="mapping", hue_order=order, aspect=1.61)

		fg.map(plt.scatter, x, y).add_legend()


	def histo(self, dataset=None, channels=None):
		if dataset  is None: dataset=self.dataset
		if channels is None: channels=self.channels

	def log(self, i):
		if i > 0: return log(i)
		else: return None

	def plot(self, dataset=None, xfunc=None, yfunc=None, transpose=False, save=False, **kwargs):
		# We are using the plot method associated with the dataframe
		# Note that that transposing the 
		if not dataset: dataset = self.dataset 
		if transpose: 
			temp = kwargs["x"]
			kwargs["x"] = kwargs["y"]
			kwargs["y"] = temp
		if xfunc: dataset[kwargs["x"]] = dataset[kwargs["x"]].apply(func=lambda x: xfunc(x))
		if yfunc: dataset[kwargs["y"]] = dataset[kwargs["y"]].apply(func=lambda y: yfunc(y))
		self.dataset.plot(**kwargs)
		if not save: plt.show()

	def freq(self, column, dataset=None, scope = 64, func=None, *args, **kwargs):
		print("[FREQ] Running frequency method")
		if not dataset: dataset=self.dataset
		if func: dataset[column] = dataset[column].apply(func=lambda x: func(x))

		# Converting it to a more 
		_min, _max = dataset[column].min(), dataset[column].max()
		res   = (_max - _min)/scope
		frequency = dataset[column].groupby(cut(dataset[column], np.arange(_min,_max,res))).count()

		return frequency

	def saveplots(self, func=None, folder=None, rdata=False, delimiter="\\", description=None, *args, **kwargs):
		if not folder: folder=self.config["OUTPUT"]

		use("Agg")
		for pos, file in enumerate(self.files): 
			self.read(file=file)
			name = f"{folder}{self.names[pos].replace('.fcs','')}{description}{pos}.png"

			if not func: 
				dataset.plot(**kwargs)

			elif func and not rdata: 
				func(*args, **kwargs)

			elif func and rdata:
				data = func(*args, **kwargs)
				data.plot()
			plt.savefig(name)
			plt.clf()

def _log(i):
	if i > 0: return log(i)
	else: return None


if __name__ == '__main__':
	use("Agg")
	run = Analyze()
	run.read()

	# run.kmeans(channels=["FSC-A", "SSC-A"], logx=False, logy=True, transpose=False, nclusters=3)
	# run.plot(x="FSC-A", y="SSC-A",  kind="scatter", transpose=False)
	# run.plot(x="FSC-A", y="SSC-A", yfunc=_log, kind="scatter", transpose=False)
	# freq = run.freq("FSC-A", scope=500)
	# run.plot(x="FSC-A", y="SSC-A", yfunc=_log, kind="scatter", transpose=False, save=True)
	# plt.savefig("blah.png")
	# print(freq[])


	# run.plot(x=freq[0], y=run.dataset["FSC-A"], kind="scatter")
	# run.saveplots(run.freq, column="FSC-A", scope=500, rdata=True, delimiter='\\')

	# run.saveplots(run.plot, x="FSC-A", y="SSC-A", yfunc=_log, kind="scatter", transpose=False, save=True, description="FSC-A_ON_SSC-A")
	# run.saveplots(run.kmeans, channels=["FSC-A", "SSC-A"], logx=False, logy=True, transpose=False, nclusters=3, description="kmeans")





