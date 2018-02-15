# Reader of fcs files
# (cc) 2017 Ali Rassolie
# Formagna

from pandas import DataFrame as df
import fcsparser
import numpy as np

__version__ = "0.1"


class fcsReader:
	def __init__(self, path, **kwargs):
	
		self.path = path
		# Returns the data contained in the fcs file
	
		self.meta, self.data = fcsparser.parse(path, 
									meta_data_only = False,
								 	reformat_meta=True)

	def rdata(self, path=None, **kwargs):
	# Returns the data retrieved from the .fcs file, which
	# in turn is related to different channels. Note that
	# the it is returned in a pandas.dataframe type. 
		if self.path: path = self.path
		elif path: self.path = path
		else: raise "Path to .fcs file has not been provided"
		
		self.meta, self.data = fcsparser.parse(path, 
							meta_data_only = False,
						 	reformat_meta=True)
		return self.data

	def rmeta(self, path=None, **kwargs):
	# This method returns the metadata of the fcs file
	# such as the various channels which are necessary
	# for the later analysis. Refer to the README for further
	# detail regarding the different channels involved. 
		if self.path: path = self.path
		elif path: self.path = path
		else: raise "Path to .fcs file has not been provided"

		self.meta, self.data = fcsparser.parse(path, 
								meta_data_only = False,
								reformat_meta=True)
		return self.meta

	def save_data(self):
		pass


		