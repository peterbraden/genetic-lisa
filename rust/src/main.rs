extern crate darwin_rs;
extern crate rand;
extern crate jpeg_decoder;
extern crate serde;
extern crate serde_json;
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate serde_derive;

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

#[derive(Debug, Clone, Serialize, Deserialize)]
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
        let ctx = &LISA;
		let mut out = String::new();
		let mut fill = String::new();
		let cx = self.x * ctx.width as f64;
		let cy = self.y * ctx.height as f64;
		let rad = self.rad * ctx.width as f64;
		write!(&mut fill, "rgba({},{},{},{})", r(self.color), g(self.color), b(self.color), self.opacity)
			.expect("String concat failed");
		write!(&mut out, "<circle cx='{}' cy='{}' r='{}' fill='{}' />", cx, cy, rad, fill)
			.expect("String concat failed");
		return out;
	}
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Canvas {
	pixels: Vec<u8>
}

impl Canvas {
	pub fn new() -> Canvas {
		let mut vec =Vec::with_capacity(LISA.image.len());
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
		let mut total = 0;
		for x in 0..LISA.image.len() {
			let diff = LISA.image[x] as i32 - self.pixels[x] as i32;
			total += diff * diff;
		}
		return total as f64;
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
    pub fn create_with(data: Vec<Circle>) -> Lisa {
        let mut res = Lisa {
            circles: data,
            canv: Canvas::new()
        };
        res.draw();
        return res;
    }

	pub fn svg(&self) -> String {
		let mut out = String::new();
		let mut contents = String::new();
		for c in &self.circles {
			contents.push_str(&c.svg());
		}
        let svgprelude = "svg xmlns='http://www.w3.org/2000/svg' style='background-color: #000;'";
		write!(&mut out, "<{} viewBox='0 0 {} {}' >{}</svg>",
                svgprelude, LISA.width, LISA.height, contents)
                .expect("String concat failed");
		return out;
	}

	pub fn draw(&mut self) {
		self.canv.wipe();
		for c in &self.circles {
			self.canv.draw(c);
		}
	}
}

impl Individual for Lisa {
    fn mutate(&mut self) {
		if rand() < 0.2 {
			self.circles.push(Circle::random());
		}

        // remove random elements
		if rand() < 0.2 && self.circles.len() > 0 {
            self.circles.retain(|_| rand() > 0.1 );
		}

        match rand::thread_rng().choose_mut(&mut self.circles) {
            Some(m) => {
                if rand() < 0.2 {
                    m.color = m.color.saturating_add(((rand() - 0.5) * 10.) as u32);
                }
                if rand() < 0.2 {
                    m.x += (rand() - 0.5) * 0.01;
                }
                if rand() < 0.2 {
                    m.y += (rand() - 0.5) * 0.01;
                }
                if rand() < 0.2 {
                    m.opacity += (rand() - 0.5) * 0.01;
                }
                if rand() < 0.2 {
                    m.rad += (rand() - 0.5) * 0.01;
                }
            },
            None => {}
        }

		self.draw() ;
    }

    fn calculate_fitness(&mut self) -> f64 {
		let fitness = self.canv.diff() * (1. + 0.001 * (self.circles.len() as f64));
        // Pixel difference * 100% + 0.1% per circle
		//print!(". {}\n", fitness);
		return fitness;
    }

    fn reset(&mut self) {
		self.circles = Vec::new();
		self.canv.wipe();
    }

	fn new_fittest_found(&mut self) {
		print!("\nNew fittest: {:?} {}", self.calculate_fitness(), self.circles.len());
        let mut svg = File::create("best.svg").unwrap();
        std::io::Write::write_all(&mut svg, self.svg().as_bytes()).expect("couldn't write");

        let mut jsonfile = File::create("best.json").unwrap();
        std::io::Write::write_all(&mut jsonfile,
                serde_json::to_string(&self.circles).expect("Serialize error").as_bytes()
            ).expect("couldn't write json");
    }
}

fn make_population(count: u32) -> Vec<Lisa> {
	let mut result = Vec::new();

    for _ in 0..count {
        let mut x = Lisa::new();
        x.mutate();
        result.push(x);
	}
	return result;
}

fn make_population_from_file(count: u32, path: &str) -> Vec<Lisa> {
	let mut result = Vec::new();
    let f = File::open(path).unwrap();
    let saved: Vec<Circle> = serde_json::from_reader(f).unwrap();

    for _ in 0..count {
        let mut x = Lisa::create_with(saved.clone());
        x.mutate();
        result.push(x);
	}
	return result;
}

fn main() {
	println!("Loaded source image {}x{}", LISA.width, LISA.height);
	//let my_pop = make_population(100);
	let my_pop = make_population_from_file(100, "best.json");
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
		.fitness(0.0)
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
