# Reader of fcs files
# (cc) 2017 Ali Rassolie
# Formagna

from pandas import DataFrame as df
import fcsparser
import numpy as np

class fcsReader:
	def __init__(self, path, **kwargs):
		self.path = path

	def data(self, path=None, **kwargs):
		if self.path: path = self.path
		elif path: self.path = path
		else: 
			raise "Path to .fcs file has not been provided"

		# Returns the data contained in the fcs file
		meta, data = fcsparser.parse(path, 
									meta_data_only = False,
								 	reformat_meta=True)
		return data