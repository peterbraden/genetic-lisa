// 
// Simple Raster Canvas implementation for fast in-memory
// drawing and comparison.
//
extern crate rand;

use std::fmt::Write;
use rand::Rng;

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Circle {
	pub x: f64,
	pub y: f64,
    pub r: u8,
    pub g: u8,
    pub b: u8,
	pub rad: f64,
	pub opacity: f64,
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

    pub fn mutate(&mut self) {
        match (rand() * 10.) as u8 {
            0...1 => self.r = rand_color_adjust(self.r),
            1...2 => self.g = rand_color_adjust(self.g),
            2...3 => self.b = rand_color_adjust(self.b),
            3...4 => self.opacity = rand_adjust(self.opacity, 0.1, 0., 1.0),
            4...6 => self.x += rand_adjust(self.x, 0.5, 0., 1.0),
            6...8 => self.y += rand_adjust(self.y, 0.5, 0., 1.0),
            8...10 => self.rad += rand_adjust(self.rad, 0.5, 0.01, 1.0),
            _ => panic!()
        }
    }

	pub fn svg(&self, width: usize, height: usize) -> String {
		let mut out = String::new();
		let mut fill = String::new();
		let cx = (self.x * width as f64) as i32;
		let cy = (self.y * height as f64) as i32;
		let rad = (self.rad * width as f64) as i32;
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
pub struct Canvas {
    width: usize,
    height: usize,
    depth: usize,
	pixels: Vec<u8>
}

impl Canvas {
	pub fn new(width: usize, height: usize, depth: usize) -> Canvas {
        let len = width * height * depth;
		let mut vec = Vec::with_capacity(len);
		for _ in 0..len {
			vec.push(0)
		}
		Canvas {
            width: width,
            height: height,
            depth: depth,
			pixels: vec
		}
	}

    pub fn from(width: usize, height:usize, depth:usize, data: Vec<u8>) -> Canvas {
		Canvas {
            width: width,
            height: height,
            depth: depth,
			pixels: data
		}
    }

   // pub fn len(&self) -> usize {
   //     self.pixels.len()
   // }

	pub fn wipe(&mut self) {
		for x in 0..self.pixels.len() {
			self.pixels[x] = 0;
		}
	}

	pub fn draw(&mut self, c: &Circle) {
        let rad = (c.rad * self.width as f64) as i32;
        let cx = (c.x * self.width as f64) as i32;
        let cy = (c.y * self.height as f64) as i32;
		let radrad = rad * rad;

		for x in -rad .. rad {
			for y in -rad .. rad {
                if x*x + y*y <= radrad {
                    let px = cx + x;
                    let py = cy + y;
                    if px >= 0 && px < self.width as i32 &&
                       py >= 0 && py < self.height as i32 {
                        self.add_pixel(px, py, c.r, c.g, c.b, c.opacity);
                    }
                }

			}
		}
	}

    pub fn add_pixel(&mut self, x: i32, y: i32, r: u8, g:u8, b:u8, opacity:f64) {
        let i = ((y * self.width as i32 + x) * self.depth as i32) as usize;
        self.pixels[i]     = color_add(self.pixels[i],      r, opacity);
        self.pixels[i + 1] = color_add(self.pixels[i + 1],  g, opacity);
        self.pixels[i + 2] = color_add(self.pixels[i + 2],  b, opacity);
    }

	pub fn diff(&self, canv: &Canvas) -> f64 {
		let mut total = 0.;
		for x in 0..canv.pixels.len() {
			let pixdiff = canv.pixels[x] as i32 - self.pixels[x] as i32;
			total += (pixdiff * pixdiff) as f64;
		}
		return total;
	}


    /*
    pub fn save(&self) {
        image::save_buffer(&Path::new("best.png"), 
            &self.pixels.as_slice(), 
            self.width as u32, 
            self.height as u32, image::RGB(8)).unwrap()
    }
    */
}
