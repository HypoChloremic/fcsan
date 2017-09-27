# class C:
#   def __init__(self):
#   	self._x = 3

#   def getx(self): 
#   	print(self._x)
#   	return self._x
  
#   def setx(self, value): 
#   	self._x = 3

#   def delx(self):
#    del self._x

#   x = property(getx, setx, delx, "I'm the 'x' property.")

class Center():
	def __init__(self):
		self.queue = None
		self._va = 0
		print(2323)

	@property
	def va(self):
		print(231)
		return self._va

	@va.setter
	def va(self, value):
		self._va = value
		print("sdsd")
		self.whenChanged()

	def whenChanged(self):
		print("this was ")
		next = self.queue.dequeue()
		next.function()


if __name__ == '__main__':
	run = Center()
	run._va = 2


