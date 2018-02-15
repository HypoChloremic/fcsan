# Test of fcsreader.py
# (cc) 2017 Ali Rassolie
# Formagna


import fcsreader
import matplotlib.pyplot as plt
process = fcsreader.fcsReader("data.fcs")
s = process.data()
print(s.columns)
s.plot(x = "SSC-A", y="FSC-A", kind="scatter")
plt.show()