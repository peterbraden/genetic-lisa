// Ideas
// - Fitness weighting - weight important areas of the source image (faces etc) more heavily in the
// fitness function.
// - Different shapes
// - Compare the gzipped SVG length to the source image length

extern crate darwin_rs;
extern crate jpeg_decoder;
extern crate serde;
extern crate serde_json;
extern crate env_logger;
#[macro_use] extern crate serde_derive;
#[macro_use] extern crate clap;
pub mod rando;
pub mod canvas;
pub mod color;

use jpeg_decoder::Decoder;
use std::fs::File;
use std::io::BufReader;
use std::fmt::Write;
use darwin_rs::{Individual, SimulationBuilder, PopulationBuilder};
use clap::{Arg, App};
use std::sync::Arc;
use canvas::{Canvas, Shape, Triangle, Circle};
use rando::{rand, choose};

#[derive(Debug)]
struct Context {
	image: Canvas,
    weightings: Canvas,
	width: i32,
	height: i32,
    depth: i32,
    format: jpeg_decoder::PixelFormat,
    mutations: u64,
    shape_multiplier: f64 // 50 = Circles only, 100 = Circles  + triangles
}

impl Context {
	pub fn new(name:&str) -> Context{
		let f = File::open(name).expect("failed to open file");
		let buf = BufReader::new(f);
		let mut jpg = Decoder::new(buf);
		let image = jpg.decode().expect("failed to decode image");
		let meta = jpg.info().unwrap();

		return Context {
			image: Canvas::from(meta.width as usize, meta.height as usize, 3, image),
            weightings: Canvas::new(meta.width as usize, meta.height as usize, 1),
			height: meta.height as i32,
			width: meta.width as i32,
            depth: 3,
            format: meta.pixel_format,
            mutations: 0,
            shape_multiplier:100.
		};
	}

    pub fn weight_entropy(&mut self) {
        // Copy image
        for i in 0..self.image.len() {
            self.weightings.pixels[i] = self.image.pixels[i];
        }
        
        self.weightings.save("weighting.png");
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializedLisa {
	circles: Vec<Circle>,
	triangles: Vec<Triangle>,
    mutations: u64,
    mutation_appends: u64,
    mutation_pops: u64,
    mutation_merges: u64,
    mutation_changes: u64,
}

#[derive(Debug, Clone)]
struct Lisa {
	circles: Vec<Circle>,
	triangles: Vec<Triangle>,
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
			triangles: Vec::new(),
			canv: Canvas::new(ctx.width as usize, 
                              ctx.height as usize, 
                              ctx.depth as usize),
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
            triangles: data.triangles,
            canv: Canvas::new(ctx.width as usize, 
                              ctx.height as usize, 
                              ctx.depth as usize),
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
			contents.push_str(&c.svg(self.ctx.width as usize, self.ctx.height as usize));
		}
		for c in &self.triangles {
			contents.push_str(&c.svg(self.ctx.width as usize, self.ctx.height as usize));
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
            c.draw_onto(&mut self.canv);
		}
		for t in &self.triangles {
            t.draw_onto(&mut self.canv);
		}
	}

    // Assumes a len
    fn remove_random_circle(&mut self) -> Circle {
        let i = (rand() * self.circles.len() as f64) as usize;
        return self.circles.remove(i);
    }

    fn remove_random_triangle(&mut self) -> Triangle {
        let i = (rand() * self.triangles.len() as f64) as usize;
        return self.triangles.remove(i);
    }

    fn add_circle(&mut self) { 
        self.circles.push(Circle::random());
        self.mutation_appends += 1;
    }

    fn add_triangle(&mut self) { 
        self.triangles.push(Triangle::random());
        self.mutation_appends += 1;
    }

    fn remove_circle(&mut self) {
        if self.circles.len() > 1 {
            self.remove_random_circle();
            self.mutation_pops += 1;
        }
    }

    fn remove_triangle(&mut self) {
        if self.triangles.len() > 1 {
            self.remove_random_triangle();
            self.mutation_pops += 1;
        }
    }

    fn mutate_circle(&mut self) {
        match choose(&mut self.circles) {
            Some(m) => {
                m.mutate();
                self.mutation_changes += 1;
            },
            None => {}
        }
    }

    fn mutate_triangle(&mut self) {
        match choose(&mut self.triangles) {
            Some(m) => {
                m.mutate();
                self.mutation_changes += 1;
            },
            None => {}
        }
    }

    /*
    fn merge_circles(&mut self) {
        if self.circles.len() > 2 {
            let d = self.remove_random();
            match choose(&mut self.circles) {
                Some(m) => {
                    m.merge(d);
                    self.mutation_merges += 1;
                },
                None => panic!()
            }
        }
    }
    */

    fn serialize(&mut self) -> SerializedLisa {
        SerializedLisa {
            circles: self.circles.clone(),
            triangles: self.triangles.clone(),
            mutations: self.mutations,
            mutation_appends: self.mutation_appends,
            mutation_pops: self.mutation_pops,
            mutation_merges: self.mutation_merges,
            mutation_changes: self.mutation_changes,
        }
    }

    fn str(&mut self) -> String {
		let mut out = String::new();
		write!(&mut out, "[F:{:.1} ({} circ, {} tri, {} mut: {}+ {}- {}~ )]",
            self.calculate_fitness(), self.circles.len(), self.triangles.len(), self.mutations,
            self.mutation_appends, self.mutation_pops, self.mutation_changes
            ).expect("couldn't append string");
        return out;
    }

}

impl Individual for Lisa {

    fn mutate(&mut self) {
        match (rand() * self.ctx.shape_multiplier) as u8 {
            0...15 => self.add_circle(),
            15...30 => self.remove_circle(),
            30...50 => self.mutate_circle(),

            50...60 => self.add_triangle(),
            60...70 => self.remove_triangle(),
            70...100 => self.mutate_triangle(),
            _ => panic!("Impossible mutation")
        }
        self.mutations += 1;
    }

    fn calculate_fitness(&mut self) -> f64 {
		self.draw();
		let fitness = self.canv.diff(&self.ctx.image) * (1. + 0.001 * (self.circles.len() as f64));
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
        //self.canv.save();
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
        .threads(1)
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
