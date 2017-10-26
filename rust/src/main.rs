// Ideas
// fitness function.
// - Different shapes
// - Compare the gzipped SVG length to the source image length
// - Divide and conquer by splitting image into small cells.


#![cfg_attr(feature="clippy", feature(plugin))]
#![cfg_attr(feature="clippy", plugin(clippy))]

extern crate darwin_rs;
extern crate serde;
extern crate serde_json;
extern crate env_logger;
extern crate chrono;
extern crate jpeg_decoder;

#[macro_use] extern crate serde_derive;
#[macro_use] extern crate clap;

pub mod rando;
pub mod canvas;
pub mod color;
pub mod shapes;
pub mod shapelist;
pub mod lisa;
pub mod context;
pub mod canvascache;

use darwin_rs::{Individual, SimulationBuilder, PopulationBuilder};
use clap::{Arg, App};
use lisa::Lisa;
use context::Context;
use std::sync::Arc;

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
                 .arg(Arg::with_name("no-triangles")
                      .long("xt"))
                 .arg(Arg::with_name("no-circles")
                      .long("xc"))
                 .arg(Arg::with_name("no-rects")
                      .long("xr"))
                 .get_matches();

    let population = value_t!(matches.value_of("population"), usize).unwrap_or(3);
    let growth = value_t!(matches.value_of("exponential-growth"), f64).unwrap_or(1.0);
    let start_with_best = matches.is_present("loadbest"); 
    let use_weighting = matches.is_present("weighting"); 
    let image = value_t!(matches.value_of("image"), String).unwrap_or(String::from("lisa.jpg"));

    let use_triangles = !matches.is_present("no-triangles");
    let use_circles = !matches.is_present("no-circles");
    let use_rectangles = !matches.is_present("no-rects");

    let mut context = Context::new(&image, use_weighting, use_triangles, use_circles, use_rectangles);

    if context.use_weighting {
        context.weight_entropy();
    }
    let ctx = Arc::new(context);
    let mut my_pop;

	println!("# Loaded source image {}x{} {:?}", ctx.width, ctx.height, ctx.format);
    println!("# Using T:{} C:{} R:{}", ctx.use_triangles, ctx.use_circles, ctx.use_rectangles);
    if start_with_best {
        my_pop = Lisa::make_population_from_file(population, ctx.clone(), "best.json");
        println!("# - Using previous best: {:.1}, {}",
                 my_pop[0].calculate_fitness(), my_pop[0].shapes.len());
        ctx.cache.lock().unwrap().insert(&my_pop[0].shapes);
    } else {
	    my_pop = Lisa::make_population(population, ctx);
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
