extern crate serde_json;
extern crate chrono;
extern crate jpeg_decoder;

use shapelist::{ShapeList};
use rando::{rand};

use std::sync::Arc;
use darwin_rs::{Individual};
use std::fs::File;
use std::fmt::Write;
use context::Context;
use std;


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedLisa {
	shapes: ShapeList,
    mutations: u64,
    mutation_appends: u64,
    mutation_pops: u64,
    mutation_merges: u64,
    mutation_changes: u64,
}

#[derive(Debug, Clone)]
pub struct Lisa {
	pub shapes: ShapeList,
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
            mutations: 0,
            mutation_appends: 0,
            mutation_pops: 0,
            mutation_merges: 0,
            mutation_changes: 0,
            ctx: ctx
		}
	}

    pub fn make_population(count: usize, ctx: Arc<Context>) -> Vec<Lisa> {
        let mut result = Vec::new();

        for _ in 0..count {
            let mut x = Lisa::new(ctx.clone());
            x.mutate();
            result.push(x);
        }
        return result;
    }

    pub fn make_population_from_file(count: usize, ctx: Arc<Context>, path: &str) -> Vec<Lisa> {
        let mut result = Vec::new();
        let f = File::open(path).unwrap();
        let saved: SerializedLisa = serde_json::from_reader(f).unwrap();

        for _ in 0..count {
            result.push(Lisa::create_with(ctx.clone(), saved.clone()));
        }
        return result;
    }

    pub fn create_with(ctx:Arc<Context>, data: SerializedLisa) -> Lisa {
        let res = Lisa {
            shapes: data.shapes,
            mutations: data.mutations,
            mutation_appends: data.mutation_appends,
            mutation_pops: data.mutation_pops,
            mutation_merges: data.mutation_merges,
            mutation_changes: data.mutation_changes,
            ctx: ctx
        };
        return res;
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
        let ctx = Arc::clone(&mut self.ctx);
        let mut cache = ctx.cache.lock().unwrap();
		let canv = cache.canvas_for(&self.shapes);
        
        if ctx.use_weighting {
            // Pixel difference * 100% + 0.1% per circle
            let fitness = canv.weighted_diff(&ctx.image, &ctx.weightings, 0.01);
		    return fitness * (1. + 0.001 * (self.shapes.len() as f64));
        } else {
            let fitness = canv.diff(&ctx.image);
		    return fitness * (1. + 0.001 * (self.shapes.len() as f64));
        }
    }

    fn reset(&mut self) {
		self.shapes = ShapeList::new();
    }

	fn new_fittest_found(&mut self) {
        let now = chrono::Utc::now();
		print!("{} New fittest: {} \n", now, self.str());
        self.ctx.cache.lock().unwrap().insert(&self.shapes);
        let mut svg = File::create("best.svg").unwrap();
        std::io::Write::write_all(&mut svg, self.svg().as_bytes()).expect("couldn't write");

        let mut jsonfile = File::create("best.json").unwrap();
        std::io::Write::write_all(&mut jsonfile,
                serde_json::to_string(&self.serialize()).expect("Serialize error").as_bytes()
            ).expect("couldn't write json");
    }
}

