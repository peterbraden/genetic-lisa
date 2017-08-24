// 
// Simple Raster Canvas implementation for fast in-memory
// drawing and comparison.
//
//extern crate image;
use std::fmt::Write;
use color::Color;
use rando::{rand, rand_adjust, randu8};


fn color_add(c:u8, c2: u8, opacity: f64) -> u8 {
	return (c as f64 * (1. - opacity) + (c2 as f64 * opacity)).min(255.).max(0.) as u8;
}

pub trait Shape {
    fn random() -> Self;
    fn mutate(&mut self);
    fn svg(&self) -> String;
    fn draw_onto(&self, &mut Canvas);
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Circle {
	pub x: f64,
	pub y: f64,
	pub rad: f64,
    pub color: Color
}

impl Circle {
	pub fn random() -> Circle {
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

    pub fn mutate(&mut self) {
        match (rand() * 10.) as u8 {
            0...4 => self.color = self.color.mutate(),
            4...6 => self.x += rand_adjust(self.x, 0.5, 0., 1.0),
            6...8 => self.y += rand_adjust(self.y, 0.5, 0., 1.0),
            8...10 => self.rad += rand_adjust(self.rad, 0.5, 0.01, 1.0),
            _ => panic!()
        }
    }

    pub fn merge(&mut self, d: Circle) {
        self.color = (&self.color + &d.color) * 0.5;
        self.x = (self.x + d.x) / 2.;
        self.y = (self.y + d.y) / 2.;
        self.rad = (self.rad + d.rad) / 2.;
    }

	pub fn svg(&self, width: usize, height: usize) -> String {
		let mut out = String::new();
		let cx = (self.x * width as f64) as i32;
		let cy = (self.y * height as f64) as i32;
		let rad = (self.rad * width as f64) as i32;
		write!(&mut out, "<circle cx='{}' cy='{}' r='{}' fill='{}' />",
                cx, cy, rad, self.color.rgba())
			.expect("String concat failed");
		return out;
	}

    pub fn draw_onto(&self, mut canvas: &mut Canvas) {
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
        self.pixels[i]     = color_add(self.pixels[i],      color.r, color.opacity);
        self.pixels[i + 1] = color_add(self.pixels[i + 1],  color.g, color.opacity);
        self.pixels[i + 2] = color_add(self.pixels[i + 2],  color.b, color.opacity);
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


    /*
    pub fn save(&self) {
        image::save_buffer(&Path::new("best.png"), 
            &self.pixels.as_slice(), 
            self.width as u32, 
            self.height as u32, image::RGB(8)).unwrap()
    }
    */
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
