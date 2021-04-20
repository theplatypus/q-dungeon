class Edge : 

	def __init__(self, A, B, label = "", is_passable = True) :
		self.A = A 
		self.B = B
		self.label = label
		self.is_passable = is_passable
		self.is_directed = False

	def __str__(self) :
		return str("%s%s%s[%s]" % (self.A, "->" if self.is_directed else "<->", self.B, self.is_passable))
