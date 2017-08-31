extern crate jpeg_decoder;

use color::Color;
use jpeg_decoder::Decoder;
use std::fs::File;
use std::cmp::{min};
use canvas::{Canvas};
use std::io::BufReader;
use canvascache::CanvasCache;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub struct Context {
	pub image: Canvas,
    pub weightings: Canvas,
	pub width: i32,
	pub height: i32,
    pub depth: i32,
    pub format: jpeg_decoder::PixelFormat,
    pub mutations: u64,
    pub use_weighting: bool,
    pub cache: Arc<Mutex<CanvasCache>>
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
            use_weighting: use_weighting,
            cache: Arc::new(Mutex::new(CanvasCache::new(meta.width as usize, meta.height as usize, 3)))
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

