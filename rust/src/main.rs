extern crate darwin_rs;
extern crate rand;
extern crate jpeg_decoder;
extern crate serde_derive;
extern crate serde;

#[macro_use]
extern crate lazy_static;
extern crate serde_json;

use jpeg_decoder::Decoder;
use std::fs::File;
use std::io::BufReader;
use std::fmt::Write;

use darwin_rs::{Individual, SimulationBuilder, PopulationBuilder};
use rand::Rng;

const PIXELDEPTH:i32 = 3;

lazy_static! {
	static ref LISA: Context = Context::new("lisa.jpg");
}

fn coord(x: i32, y: i32) -> i32 {
	return (x * LISA.width as i32 + y) * PIXELDEPTH;
}

fn rand() -> f64 {
	return rand::thread_rng().gen_range(0.,1.);
}
fn rand_color() -> u32 {
	return rand::thread_rng().gen();
}

fn from_rgb(r:u32, g:u32, b: u32) -> u32 {
	return (r << 16) + (g << 8) + b;
}

fn color_add(c:u32, c2: u32, opacity: f64) -> u32 {
	return c + (c2 as f64 * opacity) as u32;
}

fn r(c:u32) -> u8 {
	(c >> 16 % 256) as u8
}
fn g(c:u32) -> u8 {
	(c >> 8 % 256) as u8
}
fn b(c:u32) -> u8 {
	(c % 256) as u8
}

struct Context {
	image: Vec<u8>,
	width: u16,
	height: u16
}

impl Context {
	pub fn new(name:&str) -> Context{
		let f = File::open(name).expect("failed to open file");
		let buf = BufReader::new(f);
		let mut jpg = Decoder::new(buf);
		let image = jpg.decode().expect("failed to decode image");
		let meta = jpg.info().unwrap();

		return Context {
			image: image,
			height: meta.height,
			width: meta.width
		};
	}
}

#[derive(Debug, Clone)]
struct Circle {
	x: f64,
	y: f64,
	rad: f64,
	color: u32,
	opacity: f64,
}

impl Circle {
	pub fn random() -> Circle {
		Circle {
			x: rand(),
			y: rand(),
			rad: rand(),
			color: rand_color(),
			opacity: rand()
		}
	}

	pub fn svg(&self) -> String {
		let mut out = String::new();
		let mut fill = String::new();
		let cx = self.x * LISA.width as f64;
		let cy = self.y * LISA.height as f64;
		let r = self.rad * LISA.width as f64;
		write!(&mut fill, "rgba({},{},{},{})", r(self.color), g(self.color), b(self.color), self.opacity)
			.expect("String concat failed");
		write!(&mut out, "<circle cx='{}' cy='{}' r='{}' fill='{}' />", cx, cy, r, fill)
			.expect("String concat failed");
		return out;
	}
}

#[derive(Debug, Clone)]
struct Canvas {
	pixels: Vec<u8>
}

impl Canvas {
	pub fn new() -> Canvas {
		let mut vec =Vec::with_capacity((LISA.width * LISA.height) as usize);
		for _ in LISA.image.iter() {
			vec.push(0)
		}
		Canvas {
			pixels: vec
		}
	}

	pub fn wipe(&mut self) {
		for x in 0..self.pixels.len() {
			self.pixels[x] = 0;
		}
	}

	pub fn draw(&mut self, c: &Circle) {
		let xmin = - c.rad * LISA.width as f64;
		let xmax = c.rad * LISA.width as f64;
		let radrad = (c.rad * LISA.width as f64) * (c.rad * LISA.width as f64);

		for x in (xmin as i32)..(xmax as i32) {
			let height = (radrad - (x as f64 * x as f64)).sqrt();
			for y in (-height as i32)..(height as i32) {
				let i = coord(x + (c.x * LISA.width as f64) as i32, y + (c.y * LISA.height as f64) as i32) as usize;

				if x as f64 + c.x > 0. && y as f64 + c.y > 0. && i < (LISA.width * LISA.height) as usize {
					let color = from_rgb(self.pixels[i] as u32, self.pixels[i+1] as u32, self.pixels[i+2] as u32);
					let newcolor = color_add(color, c.color, c.opacity);
					self.pixels[i] = r(newcolor);
					self.pixels[i + 1] = g(newcolor);
					self.pixels[i + 2] = b(newcolor);
				}
			}
		}
	}

	pub fn diff(&self) -> f64 {
		let mut total = 0.;
		for x in 0..LISA.image.len() {
			let diff = LISA.image[x] as f64 - self.pixels[x] as f64;
			total += diff * diff;
		}
		return total;
	}
}



#[derive(Debug, Clone, Serialize, Deserialize)]
struct Lisa {
	circles: Vec<Circle>,
	canv: Canvas
}

impl Lisa {
	pub fn new() -> Lisa {
		Lisa {
			circles: Vec::new(),
			canv: Canvas::new()
		}
	}

	pub fn svg(&self) -> String {
		let mut out = String::new();
		let mut contents = String::new();
		for c in &self.circles {
			contents.push_str(&c.svg());
		}
		write!(&mut out, "<svg>{}</svg>", contents).expect("String concat failed");
		return out;
	}

	pub fn draw(&mut self) {
		self.canv.wipe();
		for c in &self.circles {
			self.canv.draw(c);
		}
	}

	pub fn draw_latest(&mut self) {
		// Assumes canvas has all of the circles except the last.
		self.canv.draw(self.circles.last().unwrap())
	}
}

impl Individual for Lisa {
    fn mutate(&mut self) {
		if rand() < 0.7 {
			self.circles.push(Circle::random());
			self.draw_latest();
		} else {
			// TODO remove random
			self.circles.pop();
			self.draw();
		}
    }

    fn calculate_fitness(&mut self) -> f64 {
		let fitness = self.canv.diff() * (self.circles.len() as f64 + 1.);
		print!(".");
		return fitness;
    }

    fn reset(&mut self) {
		self.circles = Vec::new();
    }

	fn new_fittest_found(&mut self) {
		print!("New fittest: {:?} {}", self.calculate_fitness(), self.svg());
    }
}

fn make_population(count: u32) -> Vec<Lisa> {
	let mut result = Vec::new();

    for _ in 0..count {
        result.push(Lisa::new());
	}
	return result;
}


fn main() {
	println!("Loaded source image {}x{}", LISA.width, LISA.height);
	let my_pop = make_population(100);
	println!("Allocated individuals");
	let population = PopulationBuilder::<Lisa>::new()
		.set_id(1)
		.initial_population(&my_pop)
		.increasing_exp_mutation_rate(1.03)
		.reset_limit_increment(100)
		.reset_limit_start(100)
		.reset_limit_end(1000)
		.finalize().unwrap();
	println!("Built population");
	let simulation = SimulationBuilder::<Lisa>::new()
		//.fitness(0.0)
		.factor(0.5)
        .threads(2)
        .add_population(population)
		.finalize();
	println!("Initialized simulation");


	match simulation {
		Err(e) => println!("unexpected error: {}", e),

		Ok(mut simulation) => {
			println!("Starting run");
			simulation.run();
			println!("finished run");
			simulation.print_fitness();
			for res in &simulation.simulation_result.fittest {
				print!("- {} {}", res.fitness, res.individual.svg());
			}
		}
	}
}
