extern crate darwin_rs;
extern crate rand;
extern crate jpeg_decoder;
extern crate serde;
extern crate serde_json;
extern crate env_logger;
#[macro_use] extern crate serde_derive;
#[macro_use] extern crate clap;
//extern crate image;

use jpeg_decoder::Decoder;
use std::fs::File;
use std::io::BufReader;
use std::fmt::Write;
use darwin_rs::{Individual, SimulationBuilder, PopulationBuilder};
use rand::Rng;
use clap::{Arg, App};
use std::sync::Arc;

fn rand() -> f64 {
	return rand::thread_rng().gen_range(0.,1.);
}

fn randu8() -> u8 {
	return rand::thread_rng().gen_range(0,255);
}

fn rand_color_adjust(c:u8) -> u8 {
	return c.saturating_add(((rand() - 0.5) * 256.0) as u8);
}

fn rand_adjust(p:f64, range: f64, min: f64, max:f64) -> f64 {
    return (p + ((rand() - 0.5) * range)).min(max).max(min);
}

fn color_add(c:u8, c2: u8, opacity: f64) -> u8 {
	return (c as f64 * (1. - opacity) + (c2 as f64 * opacity)).min(255.).max(0.) as u8;
}

#[derive(Debug)]
struct Context {
	image: Vec<u8>,
	width: i32,
	height: i32,
    depth: i32,
    format: jpeg_decoder::PixelFormat,
    mutations: u64
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
			height: meta.height as i32,
			width: meta.width as i32,
            depth: 3,
            format: meta.pixel_format,
            mutations: 0
		};
	}
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Circle {
	x: f64,
	y: f64,
    r: u8,
    g: u8,
    b: u8,
	rad: f64,
	opacity: f64,
}

impl Circle {
	pub fn random() -> Circle {
		Circle {
			x: rand(),
			y: rand(),
			rad: rand(),
            r: randu8(),
            g: randu8(),
            b: randu8(),
			opacity: rand()
		}
	}

	pub fn svg(&self, ctx: &Arc<Context>) -> String {
		let mut out = String::new();
		let mut fill = String::new();
		let cx = (self.x * ctx.width as f64) as i32;
		let cy = (self.y * ctx.height as f64) as i32;
		let rad = (self.rad * ctx.width as f64) as i32;
		write!(&mut fill, "rgba({},{},{},{:.4})",
                self.r, self.g, self.b, self.opacity)
			.expect("String concat failed");
		write!(&mut out, "<circle cx='{}' cy='{}' r='{}' fill='{}' />",
                cx, cy, rad, fill)
			.expect("String concat failed");
		return out;
	}
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Canvas {
	pixels: Vec<u8>
}

impl Canvas {
	pub fn new(len: usize) -> Canvas {
		let mut vec = Vec::with_capacity(len);
		for _ in 0..len {
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

	pub fn draw(&mut self, c: &Circle, ctx: &Arc<Context>) {
        let rad = (c.rad * ctx.width as f64) as i32;
        let cx = (c.x * ctx.width as f64) as i32;
        let cy = (c.y * ctx.height as f64) as i32;
		let radrad = rad * rad;

		for x in -rad .. rad {
			for y in -rad .. rad {
                if x*x + y*y <= radrad {
                    let px = cx + x;
                    let py = cy + y;
                    if px >= 0 && px < ctx.width && py >= 0 && py < ctx.height {
                        let i = ((py * ctx.width + px) * ctx.depth) as usize;
                        self.pixels[i]     = color_add(self.pixels[i],      c.r, c.opacity);
                        self.pixels[i + 1] = color_add(self.pixels[i + 1],  c.g, c.opacity);
                        self.pixels[i + 2] = color_add(self.pixels[i + 2],  c.b, c.opacity);
                    }
                }

			}
		}
	}

	pub fn diff(&self, ctx: &Arc<Context>) -> f64 {
		let mut total = 0.;
		for x in 0..ctx.image.len() {
			let pixdiff = ctx.image[x] as i32 - self.pixels[x] as i32;
			total += (pixdiff * pixdiff) as f64;
		}
		return total;
	}
}


#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializedLisa {
	circles: Vec<Circle>,
    mutations: u64,
    mutation_appends: u64,
    mutation_pops: u64,
    mutation_merges: u64,
    mutation_changes: u64,
}

#[derive(Debug, Clone)]
struct Lisa {
	circles: Vec<Circle>,
	canv: Canvas,

    mutations: u64,
    mutation_appends: u64,
    mutation_pops: u64,
    mutation_merges: u64,
    mutation_changes: u64,

    ctx: Arc<Context>
}

impl Lisa {
	pub fn new(ctx:Arc<Context>) -> Lisa {
		Lisa {
			circles: Vec::new(),
			canv: Canvas::new(ctx.image.len() as usize),
            mutations: 0,
            mutation_appends: 0,
            mutation_pops: 0,
            mutation_merges: 0,
            mutation_changes: 0,
            ctx: ctx
		}
	}
    pub fn create_with(ctx:Arc<Context>, data: SerializedLisa) -> Lisa {
        let res = Lisa {
            circles: data.circles,
            canv: Canvas::new(ctx.image.len()),
            mutations: data.mutations,
            mutation_appends: data.mutation_appends,
            mutation_pops: data.mutation_pops,
            mutation_merges: data.mutation_merges,
            mutation_changes: data.mutation_changes,
            ctx: ctx
        };
        return res;
    }

	pub fn svg(&self) -> String {
		let mut out = String::new();
		let mut contents = String::new();
		for c in &self.circles {
			contents.push_str(&c.svg(&self.ctx));
		}
        let svgprelude = "svg xmlns='http://www.w3.org/2000/svg' style='background-color: #000;' ";
		write!(&mut out, "<{} width='{}' height='{}' >{}</svg>",
                svgprelude, self.ctx.width, self.ctx.height, contents)
                .expect("String concat failed");
		return out;
	}

	pub fn draw(&mut self) {
		self.canv.wipe();
		for c in &self.circles {
			self.canv.draw(c, &self.ctx);
		}
	}

    // Assumes a len
    fn remove_random(&mut self) -> Circle {
        let i = (rand() * self.circles.len() as f64) as usize;
        return self.circles.remove(i);
    }

    fn add_circle(&mut self) { 
        self.circles.push(Circle::random());
        self.mutation_appends += 1;
    }

    fn remove_circle(&mut self) {
        if self.circles.len() > 1 {
            self.remove_random();
            self.mutation_pops += 1;
        }
    }

    fn mutate_circle(&mut self) {
        match rand::thread_rng().choose_mut(&mut self.circles) {
            Some(m) => {
                match (rand() * 10.) as u8 {
                    0...1 => m.r = rand_color_adjust(m.r),
                    1...2 => m.g = rand_color_adjust(m.g),
                    2...3 => m.b = rand_color_adjust(m.b),
                    3...4 => m.opacity = rand_adjust(m.opacity, 0.1, 0., 1.0),
                    4...6 => m.x += rand_adjust(m.opacity, 0.5, 0., 1.0),
                    6...8 => m.y += rand_adjust(m.opacity, 0.5, 0., 1.0),
                    8...10 => m.rad += rand_adjust(m.opacity, 0.5, 0.01, 1.0),
                    _ => panic!()
                }
                self.mutation_changes += 1;
            },
            None => {}
        }
    }

    fn merge_circles(&mut self) {
        if self.circles.len() > 2 {
            let d = self.remove_random();
            match rand::thread_rng().choose_mut(&mut self.circles) {
                Some(m) => {
                    m.r = ((m.r as u32 + d.r as u32) / 2) as u8;
                    m.g = ((m.g as u32 + d.g as u32) / 2) as u8;
                    m.b = ((m.b as u32 + d.b as u32) / 2) as u8;
                    m.x = (m.x + d.x) / 2.;
                    m.y = (m.y + d.y) / 2.;
                    m.rad = (m.rad + d.rad) / 2.;
                    m.opacity = m.opacity + d.opacity;

                    self.mutation_merges += 1;
                },
                None => panic!()
            }
        }
    }

    fn serialize(&mut self) -> SerializedLisa {
        SerializedLisa {
            circles: self.circles.clone(),
            mutations: self.mutations,
            mutation_appends: self.mutation_appends,
            mutation_pops: self.mutation_pops,
            mutation_merges: self.mutation_merges,
            mutation_changes: self.mutation_changes,
        }
    }

    fn str(&mut self) -> String {
		let mut out = String::new();
		write!(&mut out, "[F:{:.1} ({} circ, {} mut: {}+ {}- {}<> {}~ )]",
            self.calculate_fitness(), self.circles.len(), self.mutations,
            self.mutation_appends, self.mutation_pops, self.mutation_merges, self.mutation_changes
            ).expect("couldn't append string");
        return out;
    }

}

impl Individual for Lisa {

    fn mutate(&mut self) {
        match (rand() * 10.) as u8 {
            0...3 => self.add_circle(),
            3...4 => self.remove_circle(),
            4...6 => self.merge_circles(),
            6...10 => self.mutate_circle(),
            _ => panic!()
        }
        self.mutations += 1;
    }

    fn calculate_fitness(&mut self) -> f64 {
		self.draw();
		let fitness = self.canv.diff(&self.ctx) * (1. + 0.001 * (self.circles.len() as f64));
        // Pixel difference * 100% + 0.1% per circle
		return fitness;
    }

    fn reset(&mut self) {
		self.circles = Vec::new();
		self.canv.wipe();
    }

	fn new_fittest_found(&mut self) {
		print!("New fittest: {} \n", self.str());
        let mut svg = File::create("best.svg").unwrap();
        std::io::Write::write_all(&mut svg, self.svg().as_bytes()).expect("couldn't write");

        let mut jsonfile = File::create("best.json").unwrap();
        std::io::Write::write_all(&mut jsonfile,
                serde_json::to_string(&self.serialize()).expect("Serialize error").as_bytes()
            ).expect("couldn't write json");

        //image::save_buffer(&Path::new("best.png"), &self.canv.pixels.as_slice(), self.ctx.width as u32, self.ctx.height as u32, image::RGB(8)).unwrap()
    }
}

fn make_population(count: usize, ctx: Arc<Context>) -> Vec<Lisa> {
	let mut result = Vec::new();

    for _ in 0..count {
        let mut x = Lisa::new(ctx.clone());
        x.mutate();
        result.push(x);
	}
	return result;
}

fn make_population_from_file(count: usize, ctx: Arc<Context>, path: &str) -> Vec<Lisa> {
	let mut result = Vec::new();
    let f = File::open(path).unwrap();
    let saved: SerializedLisa = serde_json::from_reader(f).unwrap();

    for _ in 0..count {
        result.push(Lisa::create_with(ctx.clone(), saved.clone()));
	}
	return result;
}

fn main() {
    env_logger::init().expect("logger couldn't init");

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

    let ctx  = Arc::new(Context::new("lisa.jpg"));
    let population = value_t!(matches.value_of("population"), usize).unwrap_or(10);
    let growth = value_t!(matches.value_of("exponential-growth"), f64).unwrap_or(1.0);
    let start_with_best = matches.is_present("loadbest"); 
    let mut my_pop;

	println!("# Loaded source image {}x{} {:?}", ctx.width, ctx.height, ctx.format);
    if start_with_best {
        my_pop = make_population_from_file(population, ctx, "best.json");
        println!("# - Using previous best: {:.1}, {}",
                 my_pop[0].calculate_fitness(), my_pop[0].circles.len());
    } else {
	    my_pop = make_population(population, ctx);
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
