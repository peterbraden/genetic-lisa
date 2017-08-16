extern crate darwin_rs;
extern crate rand;
extern crate jpeg_decoder;
extern crate serde;
extern crate serde_json;
extern crate env_logger;
#[macro_use] extern crate lazy_static;
#[macro_use] extern crate serde_derive;
#[macro_use] extern crate clap;

use jpeg_decoder::Decoder;
use std::fs::File;
use std::io::BufReader;
use std::fmt::Write;
use darwin_rs::{Individual, SimulationBuilder, PopulationBuilder};
use rand::Rng;
use clap::{Arg, App};

const PIXELDEPTH:usize = 3;
const WHITE:u32 = (256<<16) + (256<<8) + 256;


lazy_static! {
	static ref LISA: Context = Context::new("lisa.jpg");
}

fn coord(x: i32, y: i32) -> usize {
	return ((x * LISA.width as i32 + y) * PIXELDEPTH as i32) as usize;
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
	return c.saturating_add((c2 as f64 * opacity) as u32) % WHITE;
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
	width: usize,
	height: usize,
    format: jpeg_decoder::PixelFormat
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
			height: meta.height as usize,
			width: meta.width as usize,
            format: meta.pixel_format
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
        let rad = c.rad * LISA.width as f64;
        let cx = c.x * LISA.width as f64;
        let cy = c.x * LISA.height as f64;
		let radrad = rad * rad;

		for x in (- rad as i32)..(rad as i32) {
			let height = (radrad - (x as f64 * x as f64)).sqrt();
			for y in (-height as i32)..(height as i32) {
				let i = coord(x + cx as i32, y + cy as i32);

				if x as f64 + cx > 0. &&
                   y as f64 + cy > 0. &&
                   i < (LISA.width * LISA.height * PIXELDEPTH) as usize {
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
			let pixdiff = LISA.image[x] as i32 - self.pixels[x] as i32;
			total += (pixdiff * pixdiff) as f64;
		}
		return total;
	}

    /*
    pub fn to_pixels(&self) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        for x in 0..LISA.width {
            for y in 0..LISA.height {
                let i = coord(x as i32, y as i32) as usize;
                let mut pix = self.pixels [i + 2] as u32;
                pix += (self.pixels [i + 1] as u32) << 8;
                pix += (self.pixels [i] as u32) << 16;
                out.push(pix);
            }
        }
        return out;
    }
    */
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
        let res = Lisa {
            circles: data,
            canv: Canvas::new()
        };
        return res;
    }

	pub fn svg(&self) -> String {
		let mut out = String::new();
		let mut contents = String::new();
		for c in &self.circles {
			contents.push_str(&c.svg());
		}
        let svgprelude = "svg xmlns='http://www.w3.org/2000/svg' style='background-color: #000;' ";
		write!(&mut out, "<{} width='{}' height='{}' >{}</svg>",
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
        // Add circle
		if rand() < 0.2 {
			self.circles.push(Circle::random());
		}

        // remove random elements
		if rand() < 0.2 && self.circles.len() > 0 {
            self.circles.retain(|_| rand() > 0.1 );
		}

        // Mutate random circle
        if rand() < 0.8 {
            match rand::thread_rng().choose_mut(&mut self.circles) {
                Some(m) => {
                    if rand() < 0.2 {
                        m.color = rand_color();
                    }
                    if rand() < 0.2 {
                        m.x += (rand() - 0.5) * 0.1;
                    }
                    if rand() < 0.2 {
                        m.y += (rand() - 0.5) * 0.1;
                    }
                    if rand() < 0.2 {
                        m.opacity += (rand() - 0.5) * 0.1;
                    }
                    if rand() < 0.2 {
                        m.rad += (rand() - 0.5) * 0.1;
                    }
                },
                None => {}
            }
        }

        // Merge circles?
    }

    fn calculate_fitness(&mut self) -> f64 {
		self.draw();
		let fitness = self.canv.diff() * (1. + 0.0001 * (self.circles.len() as f64));
        // Pixel difference * 100% + 0.01% per circle
		return fitness;
    }

    fn reset(&mut self) {
		self.circles = Vec::new();
		self.canv.wipe();
    }

	fn new_fittest_found(&mut self) {
		print!("\nNew fittest: {:.1} {}", self.calculate_fitness(), self.circles.len());
        let mut svg = File::create("best.svg").unwrap();
        std::io::Write::write_all(&mut svg, self.svg().as_bytes()).expect("couldn't write");

        let mut jsonfile = File::create("best.json").unwrap();
        std::io::Write::write_all(&mut jsonfile,
                serde_json::to_string(&self.circles).expect("Serialize error").as_bytes()
            ).expect("couldn't write json");
    }
}

fn make_population(count: usize) -> Vec<Lisa> {
	let mut result = Vec::new();

    for _ in 0..count {
        let mut x = Lisa::new();
        x.mutate();
        result.push(x);
	}
	return result;
}

fn make_population_from_file(count: usize, path: &str) -> Vec<Lisa> {
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
    env_logger::init();
    let matches = App::new("Lisa")
                 .arg(Arg::with_name("population")
                    .short("p")
                    .takes_value(true))
                 .arg(Arg::with_name("loadbest")
                    .short("b"))
                 .arg(Arg::with_name("exponential-growth")
                    .short("e")
                    .takes_value(true))
                 .get_matches();

    let population = value_t!(matches.value_of("population"), usize).unwrap_or(10);
    let growth = value_t!(matches.value_of("exponential-growth"), f64).unwrap_or(1.0);
    let start_with_best = matches.is_present("loadbest"); 
    let mut my_pop;

	println!("# Loaded source image {}x{} {:?}", LISA.width, LISA.height, LISA.format);
    if start_with_best {
        my_pop = make_population_from_file(population, "best.json");
        println!("# - Using previous best: {:.1}, {}",
                 my_pop[0].calculate_fitness(), my_pop[0].circles.len());
    } else {
	    my_pop = make_population(population);
    }
	println!("# Allocated individuals");
	let population = PopulationBuilder::<Lisa>::new()
		.set_id(1)
		.initial_population(&my_pop)
		.increasing_exp_mutation_rate(growth)
		.reset_limit_increment(100)
		.reset_limit_start(100)
		.reset_limit_end(0)
		.finalize().unwrap();
	println!("# Built population");
	let simulation = SimulationBuilder::<Lisa>::new()
		.fitness(0.0)
        .threads(2)
        .add_population(population)
		.finalize();
	println!("# Initialized simulation");


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
