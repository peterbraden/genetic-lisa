from PIL import Image
import aggdraw, random
#import psyco
#psyco.full()
import copy


REFERENCE = Image.open('monalisa.jpg')
REFERENCE_DATA = REFERENCE.getdata()
IMAGE_SIZE = REFERENCE.size

GENERATION_POPULATION = 10
CROSSOVER_POPULATION = 5

INITIAL_GENERATION = 3662

class Strain(object):
	"""
	"""
		
	def __init__(self, name = None, dna_string =  None, parents = None):
		self._fitness = None
		if dna_string:
			d = dna_string
			self.dna = d['dna']
			self.name = d['name']
			self._fitness = d.get('fitness')
		elif parents: #Breed randomly from parents
			self.dna = copy.deepcopy(random.choice(parents).dna)
			self.name = name
	
			if random.random() > 0.2 or len(self.dna)<20:
				self.mutate()
			
			if self.dna:
				for i in range(random.randint(1, len(parents))):
					x = random.choice(parents).dna
					y = x and random.choice(x)
					y = y and copy.deepcopy(y)
					z = random.randint(0,len(self.dna)-1)
					self.dna[z] = y or self.dna[z]
			
		else:
			self.name = name
			self.dna = []
		
	def mutate(self):
		x = random.random()
		if x<0.25 and len(self.dna)<500:
			self.dna.append({'ellipse' : generate_ellipse(), 'color' : generate_color()})
		elif x<0.75: 
			if self.dna: self.dna.pop(random.randint(0,len(self.dna)-1))
			self.dna.append({'ellipse' : generate_ellipse(), 'color' : generate_color()})
		else:
			if self.dna: self.dna.pop(random.randint(0,len(self.dna)-1))
		
		if random.random()<0.1:
			self.mutate()
		
		self._fitness = None	

		
	def __repr__(self):
		d = {'name' : self.name, 'dna' : self.dna}
		if self._fitness:
			d['fitness'] = self._fitness
		return repr(d)


	def _pix_comp(self, ref, mut):
		r = ref[0] - mut[0]
		g = ref[1] - mut[1]
		b = ref[2] - mut[2]
		return r*r + g*g + b*b

	def fitness(self):
		return self._fitness or self._fitness_func()


	def _fitness_func(self):
		"""
			Compare strain against reference image.
		"""
		
		fitness = 0
		mut = iter(REFERENCE_DATA)
		for i in self.draw().getdata():
			j = mut.next()
			fitness += self._pix_comp(i, j)
		self._fitness = fitness	
		return fitness


	def draw(self):
		draw = aggdraw.Draw('RGBA', IMAGE_SIZE, 0x0)
	
		for poly in self.dna:
			draw.ellipse(poly['ellipse'], aggdraw.Brush(poly['color']))
		draw.flush()
			
		return Image.fromstring('RGBA', IMAGE_SIZE, draw.tostring())		



def generate_polygon(im_size):
	max_points = 5
	poly = []
	
	for i in range(random.randint(3, max_points)):
		poly.append(random.randint(0, im_size[0]))
		poly.append(random.randint(0, im_size[1]))
			
	return poly


def generate_ellipse():
	x = random.randint(0, IMAGE_SIZE[0])
	y = random.randint(0, IMAGE_SIZE[1])
	width = random.randint(0, IMAGE_SIZE[0]/2)
	height = random.randint(0, IMAGE_SIZE[1]/2)
	return [x, y , x + width, y + width]

def generate_color(prev = None):
	if prev:
		return (prev[0] + random.randint(-0x3, 0x3), prev[1] + random.randint(-0x5, 0x5), prev[2] + random.randint(-0x5, 0x5), prev[3] + random.randint(-0x5, 0x5))
	return (random.randint(0x33, 0x99), random.randint(0, 0xff), random.randint(0, 0xff), random.randint(0, 0xff))



	
if __name__ == "__main__":	
	
	if INITIAL_GENERATION:
		exec(open("gen-%s.txt" % INITIAL_GENERATION, "r"))
		initial_data = [Strain(dna_string = x) for x in initial_data]
		
		i = INITIAL_GENERATION
	else:
		initial_data  = [Strain(name = 'initial') for x in range(CROSSOVER_POPULATION)]
		i=0

	best = 999999999999999999
	crossover_strains= initial_data
	last_img = best
	
	while True:
		i += 1
		
		print "\nstarting generation %s. Best of breed is %s (%s): %s " % (
			i, crossover_strains[0].name, len(crossover_strains[0].dna), crossover_strains[0].fitness())
		print "Crossover is %s" % [x.name for x in crossover_strains]
			
		
		generation = []
		for j in range(GENERATION_POPULATION):
			p = Strain(parents = crossover_strains, name = "gen%s-%s" % (i,j))
			generation.append(p)
			
		#Keep previous best of breed:
		generation.extend(crossover_strains)
		generation.sort(key = lambda x: x.fitness())
		crossover_strains = generation[:CROSSOVER_POPULATION]
		best = crossover_strains[0].fitness()
		del(generation)


		if best < last_img *0.9:
			last_img = best
			crossover_strains[0].draw().save("%s.jpg" % (crossover_strains[0].name))
			save = open("gen-%s.txt" % i, "w")
			save.write("initial_data = " + repr(crossover_strains))
			save.close()
					
