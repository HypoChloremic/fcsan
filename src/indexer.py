from analyze import Analyze
import argparse

# ap = argparse.ArgumentParser()
# ap.addargument("-f", "--folder")
# opts = ap.parse_args()

run = Analyze()
run.read()
files = run.files

def indexer():
	with open("FACS_INDEX.txt", "w") as file:
		for i in files:
			run.read(i)
			meta = run.meta
			str_to_save = f"File: {meta['$FIL']},Date: {meta['$DATE']},\n"
			file.write(str_to_save)
		

indexer()