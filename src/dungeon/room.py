
class Room : 
	
	def __init__(self, i = 1, j = 1, label = "", keypoint = None):
		self.i = i
		self.j = j
		self.label = label
		self.keypoint = keypoint
	
	def add_keypoint(self, keypoint):
		self.keypoint = keypoint

	def __repr__(self) :
		return self.__str__()
	
	def __str__(self) :
		return "%s%s" % (self.label, " [%s]" % (self.keypoint) if self.keypoint else "")

def test_room() : 
	room = Room(1, 1, "A")
	print(room)
	room.add_keypoint("Tresor")
	print(room)
	room2 = Room(2, 1)
	str(room2)
