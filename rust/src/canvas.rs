// 
// Simple Raster Canvas implementation for fast in-memory
// drawing and comparison.


extern crate image;
use std::path::Path;

use std::fmt::Write;
use color::Color;
use rando::{rand, rand_adjust, randu8};
use std::cmp::{min, max};

pub trait Shape where Self: Clone {
    fn random() -> Self;
    fn mutate(&mut self);
    fn svg(&self, width: usize, height: usize) -> String;
    fn draw_onto(&self, &mut Canvas);
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Triangle {
    pub x1: f64,
    pub x2: f64,
    pub x3: f64,
    pub y1: f64,
    pub y2: f64,
    pub y3: f64,
    pub color: Color
}

impl Shape for Triangle {
    fn random() -> Triangle {
        Triangle {
            x1: rand(),
            x2: rand(),
            x3: rand(),
            y1: rand(),
            y2: rand(),
            y3: rand(),
            color: Color {
                r: randu8(),
                g: randu8(),
                b: randu8(),
                opacity: rand()
            }
        }
    }

    fn mutate(&mut self) {
        match (rand() * 100.) as u8 {
            0...40 => self.color = self.color.mutate(),
            40...50 => self.x1 += rand_adjust(self.x1, 0.5, 0., 1.0),
            50...60 => self.y1 += rand_adjust(self.y1, 0.5, 0., 1.0),
            60...70 => self.x2 += rand_adjust(self.x2, 0.5, 0., 1.0),
            70...80 => self.y2 += rand_adjust(self.y2, 0.5, 0., 1.0),
            80...90 => self.x3 += rand_adjust(self.x3, 0.5, 0., 1.0),
            90...100 => self.y3 += rand_adjust(self.y3, 0.5, 0., 1.0),
            _ => panic!()
        }
    }

    fn svg(&self, width: usize, height: usize) -> String {
		let mut out = String::new();
		write!(&mut out, "<polygon points='{},{} {},{} {},{}' fill='{}' />",
                (self.x1 * width as f64) as i32,
                (self.y1 * height as f64) as i32,
                (self.x2 * width as f64) as i32,
                (self.y2 * height as f64) as i32,
                (self.x3 * width as f64) as i32,
                (self.y3 * height as f64) as i32,
                self.color.rgba())
			.expect("String concat failed");
		return out;
    }

    fn draw_onto(&self, canv: &mut Canvas) {
        let x1 = (self.x1 * canv.width as f64) as i32;
        let y1 = (self.y1 * canv.height as f64) as i32;
        let x2 = (self.x2 * canv.width as f64) as i32;
        let y2 = (self.y2 * canv.height as f64) as i32;
        let x3 = (self.x3 * canv.width as f64) as i32;
        let y3 = (self.y3 * canv.height as f64) as i32;
        let xmin = min(x1, min(x2, x3));
        let xmax = max(x1, max(x2, x3));
        let ymin = min(y1, min(y2, y3));
        let ymax = max(y1, max(y2, y3));
        

        for x in xmin..min(xmax, canv.width as i32) {
            for y in ymin..min(ymax, canv.height as i32) {
                let asx = x - x1;
                let asy = y - y1;
                let sab = (x2 - x1) * asy - (y2 - y1) * asx > 0;
                if ((x3 - x1) * asy - (y3 - y1) * asx > 0) == sab { continue };
                if ((x3 - x2) * (y - y2) - (y3 - y2) * (x - x2) > 0) != sab { continue };
                canv.add_pixel(x, y, &self.color)
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Circle {
	pub x: f64,
	pub y: f64,
	pub rad: f64,
    pub color: Color
}

impl Circle {
    /*
    pub fn merge(&mut self, d: Circle) {
        self.color = (&self.color + &d.color) * 0.5;
        self.x = (self.x + d.x) / 2.;
        self.y = (self.y + d.y) / 2.;
        self.rad = (self.rad + d.rad) / 2.;
    }
    */
}

impl Shape for Circle {
	fn random() -> Circle {
		Circle {
			x: rand(),
			y: rand(),
			rad: rand(),
            color: Color {
                r: randu8(),
                g: randu8(),
                b: randu8(),
                opacity: rand()
            }
		}
	}

    fn mutate(&mut self) {
        match (rand() * 10.) as u8 {
            0...4 => self.color = self.color.mutate(),
            4...6 => self.x += rand_adjust(self.x, 0.5, 0., 1.0),
            6...8 => self.y += rand_adjust(self.y, 0.5, 0., 1.0),
            8...10 => self.rad += rand_adjust(self.rad, 0.5, 0.01, 1.0),
            _ => panic!()
        }
    }


	fn svg(&self, width: usize, height: usize) -> String {
		let mut out = String::new();
		let cx = (self.x * width as f64) as i32;
		let cy = (self.y * height as f64) as i32;
		let rad = (self.rad * width as f64) as i32;
		write!(&mut out, "<circle cx='{}' cy='{}' r='{}' fill='{}' />",
                cx, cy, rad, self.color.rgba())
			.expect("String concat failed");
		return out;
	}

    fn draw_onto(&self, mut canvas: &mut Canvas) {
        let rad = (self.rad * canvas.width as f64) as i32;
        let cx = (self.x * canvas.width as f64) as i32;
        let cy = (self.y * canvas.height as f64) as i32;
		let radrad = rad * rad;

		for x in -rad .. rad {
			for y in -rad .. rad {
                if x*x + y*y <= radrad {
                    let px = cx + x;
                    let py = cy + y;
                    if px >= 0 && px < canvas.width as i32 &&
                       py >= 0 && py < canvas.height as i32 {
                        canvas.add_pixel(px, py, &self.color);
                    }
                }

			}
		}
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Canvas {
    width: usize,
    height: usize,
    depth: usize,
	pub pixels: Vec<u8>
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

   pub fn len(&self) -> usize {
        self.pixels.len()
   }

	pub fn wipe(&mut self) {
		for x in 0..self.pixels.len() {
			self.pixels[x] = 0;
		}
	}

    pub fn add_pixel(&mut self, x: i32, y: i32, color: &Color) {
        let i = ((y * self.width as i32 + x) * self.depth as i32) as usize;
        if i + self.depth < self.pixels.len() {
            color.add_to_vec(&mut self.pixels, i);
        }
    }

    /// Pixel difference squared
	pub fn diff(&self, canv: &Canvas) -> f64 {
		let mut total = 0.;
		for x in 0..canv.pixels.len() {
			let pixdiff = canv.pixels[x] as i32 - self.pixels[x] as i32;
			total += (pixdiff * pixdiff) as f64;
		}
		return total;
	}


    pub fn save(&self, filename: &str) {
        image::save_buffer(&Path::new(filename), 
            &self.pixels.as_slice(), 
            self.width as u32, 
            self.height as u32, image::RGB(8)).unwrap()
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create(){
        let c = Canvas::new(10, 10, 3);
        assert_eq!(c.len(), 300);
    }

    #[test]
    fn diff() {
        let mut c = Canvas::new(10, 10, 3);
        let c2 = Canvas::new(10, 10, 3);
        c.add_pixel(0,0, &Color {r:255,g:0,b:0,opacity:1.});
        assert_eq!(c2.diff(&c), 255. * 255.);
        assert_eq!(c.diff(&c2), 255. * 255.);
    }

    #[test]
    fn draw_shapes(){
        let mut c = Canvas::new(10, 10, 3);
        Circle::random().draw_onto(&mut c);
    }
}
