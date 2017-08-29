// Ideas
// fitness function.
// - Different shapes
// - Compare the gzipped SVG length to the source image length
// - Divide and conquer by splitting image into small cells.

extern crate darwin_rs;
extern crate jpeg_decoder;
extern crate serde;
extern crate serde_json;
extern crate env_logger;
extern crate chrono;

#[macro_use] extern crate serde_derive;
#[macro_use] extern crate clap;
pub mod rando;
pub mod canvas;
pub mod color;
pub mod shapes;
pub mod shapelist;

use jpeg_decoder::Decoder;
use std::fs::File;
use std::io::BufReader;
use std::fmt::Write;
use darwin_rs::{Individual, SimulationBuilder, PopulationBuilder};
use clap::{Arg, App};
use std::sync::Arc;
use canvas::{Canvas};
use rando::{rand};
use color::Color;
use std::cmp::{min};
use shapelist::{ShapeList};

#[derive(Debug)]
struct Context {
	image: Canvas,
    weightings: Canvas,
	width: i32,
	height: i32,
    depth: i32,
    format: jpeg_decoder::PixelFormat,
    mutations: u64,
    use_weighting: bool
}

impl Context {
	pub fn new(name:&str, use_weighting: bool) -> Context{
		let f = File::open(name).expect("failed to open file");
		let buf = BufReader::new(f);
		let mut jpg = Decoder::new(buf);
		let image = jpg.decode().expect("failed to decode image");
		let meta = jpg.info().unwrap();

		return Context {
			image: Canvas::from(meta.width as usize, meta.height as usize, 3, image),
            weightings: Canvas::new(meta.width as usize, meta.height as usize, 3),
			height: meta.height as i32,
			width: meta.width as i32,
            depth: 3,
            format: meta.pixel_format,
            mutations: 0,
            use_weighting: use_weighting
		};
	}

    pub fn weight_entropy(&mut self) {
        // Copy image
        for x in 0..self.width {
            for y in 0..self.height {
                let diff = min(((self.image.neighbors_diffsq(x, y, 1) / 10) as f64).sqrt() as i32, 255) as u8;
                self.weightings.add_pixel(x, y, &Color {
                                                    r: diff,
                                                    g: diff,
                                                    b: diff,
                                                    opacity: 1.});
            }
        }

        self.weightings.save("weighting.png");
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializedLisa {
	shapes: ShapeList,
    mutations: u64,
    mutation_appends: u64,
    mutation_pops: u64,
    mutation_merges: u64,
    mutation_changes: u64,
}

#[derive(Debug, Clone)]
struct Lisa {
	shapes: ShapeList,
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
			shapes: ShapeList::new(),
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
            shapes: data.shapes,
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


	pub fn draw(&mut self) {
        self.shapes.draw_onto(&mut self.canv);
	}

    pub fn svg(&self) -> String{
        return self.shapes.svg(self.ctx.width as usize, self.ctx.height as usize);
    }

    fn serialize(&mut self) -> SerializedLisa {
        SerializedLisa {
            shapes: self.shapes.clone(),
            mutations: self.mutations,
            mutation_appends: self.mutation_appends,
            mutation_pops: self.mutation_pops,
            mutation_merges: self.mutation_merges,
            mutation_changes: self.mutation_changes,
        }
    }

    fn str(&mut self) -> String {
		let mut out = String::new();
		write!(&mut out, "[F:{:.1} ({} shap, {} mut: {}+ {}- {}~ )]",
            self.calculate_fitness(), self.shapes.len(), self.mutations,
            self.mutation_appends, self.mutation_pops, self.mutation_changes
            ).expect("couldn't append string");
        return out;
    }

}

impl Individual for Lisa {

    fn mutate(&mut self) {
        match (rand() * 100.) as u8 {
            0...30 => {
                self.shapes.add_random();
                self.mutation_appends += 1;
                },
            30...40 => {
                self.shapes.remove_shape();
                self.mutation_pops += 1;
                },
            40...100 => {
                self.shapes.mutate();
                self.mutation_changes += 1;
                },
            _ => panic!("Impossible mutation")
        }
        self.mutations += 1;
    }

    fn calculate_fitness(&mut self) -> f64 {
		self.draw();
        
        if self.ctx.use_weighting {
            let fitness = self.canv.weighted_diff(&self.ctx.image, &self.ctx.weightings, 0.01);
		    return fitness * (1. + 0.001 * (self.shapes.len() as f64));
        } else {
            // Pixel difference * 100% + 0.1% per circle
            let fitness = self.canv.diff(&self.ctx.image);
		    return fitness * (1. + 0.001 * (self.shapes.len() as f64));
        }
    }

    fn reset(&mut self) {
		self.shapes = ShapeList::new();
		self.canv.wipe();
    }

	fn new_fittest_found(&mut self) {
        let now = chrono::Utc::now();
		print!("{} New fittest: {} \n", now, self.str());
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
                 .arg(Arg::with_name("image")
                    .short("i")
                    .takes_value(true))
                 .arg(Arg::with_name("population")
                    .short("p")
                    .takes_value(true))
                 .arg(Arg::with_name("loadbest")
                    .short("b"))
                 .arg(Arg::with_name("exponential-growth")
                    .short("e")
                    .takes_value(true))
                 .arg(Arg::with_name("weighting")
                    .short("w"))
                 .get_matches();

    let population = value_t!(matches.value_of("population"), usize).unwrap_or(10);
    let growth = value_t!(matches.value_of("exponential-growth"), f64).unwrap_or(1.0);
    let start_with_best = matches.is_present("loadbest"); 
    let use_weighting = matches.is_present("weighting"); 
    let image = value_t!(matches.value_of("image"), String).unwrap_or(String::from("lisa.jpg"));

    let mut context = Context::new(&image, use_weighting);

    if context.use_weighting {
        context.weight_entropy();
    }
    let ctx = Arc::new(context);
    let mut my_pop;

	println!("# Loaded source image {}x{} {:?}", ctx.width, ctx.height, ctx.format);
    if start_with_best {
        my_pop = make_population_from_file(population, ctx, "best.json");
        println!("# - Using previous best: {:.1}, {}",
                 my_pop[0].calculate_fitness(), my_pop[0].shapes.len());
    } else {
	    my_pop = make_population(population, ctx);
    }
	println!("# Allocated individuals: {}", population);
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
