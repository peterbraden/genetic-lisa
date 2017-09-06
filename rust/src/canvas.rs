// Simple Raster Canvas implementation for fast in-memory
// drawing and comparison.


extern crate image;
use std::path::Path;
use color::Color;
use std::cmp::{min, max};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Canvas {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
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
    
    #[inline]
    pub fn ind_from_pos(&self, x: i32, y: i32) -> i32 {
        return (y * self.width as i32 + x) * self.depth as i32;
    }

    #[inline]
    pub fn add_pixel(&mut self, x: i32, y: i32, color: &Color) {
        let i = self.ind_from_pos(x, y) as usize;
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

    pub fn weighted_diff(&self, canv: &Canvas, weights: &Canvas, scale: f64) -> f64 {
		let mut total = 0.;
		for x in 0..canv.pixels.len() {
			let pixdiff = canv.pixels[x] as i32 - self.pixels[x] as i32;
            let pixdiffsq = (pixdiff * pixdiff) as f64;
            let weighted = pixdiffsq * (1. + weights.pixels[x] as f64 * scale);
			total += weighted;
		}
		return total;
    }


    pub fn save(&self, filename: &str) {
        image::save_buffer(&Path::new(filename), 
            &self.pixels.as_slice(), 
            self.width as u32, 
            self.height as u32, image::RGB(8)).unwrap()
    }

    pub fn pixel_at(&self, x: i32, y: i32) -> Color {
        if x > 0 && x < self.width as i32 && y > 0 && y < self.height as i32 {
            let i = self.ind_from_pos(x, y) as usize;
            return Color { r: self.pixels[i],
                           g: self.pixels[i + 1],
                           b: self.pixels[i + 2],
                           opacity: 1.
            }
        } else {
            return Color::black();
        }
    }

    pub fn pixel_diff_sq(&self, x1: i32, y1:i32, x2:i32, y2:i32) -> i32{
        let p1 = self.pixel_at(x1, y1);
        let p2 = self.pixel_at(x2, y2);
        return (p1.r as i32 - p2.r as i32) * (p1.r as i32 - p2.r as i32) +
               (p1.g as i32 - p2.g as i32) * (p1.g as i32 - p2.g as i32) +
               (p1.b as i32 - p2.b as i32) * (p1.b as i32 - p2.b as i32) ;
    }


    pub fn neighbors_diffsq(&self, x: i32, y: i32, radius: i32) -> i32 {
        let mut total = 0;
        for px in max(x - radius, 0)..min(x + radius, self.width as i32){
            for py in max(y - radius, 0)..min(y + radius, self.height as i32){
                if !(px == x && py == y) {
                    total += self.pixel_diff_sq(x,y,px,py);
                }
            }
        }
        return total;
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
}
