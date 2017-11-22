extern crate jpeg_decoder;

use color::Color;
use jpeg_decoder::Decoder;
use std::fs::File;
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
    pub use_triangles: bool,
    pub use_circles: bool,
    pub use_rectangles: bool,
    pub cache: Arc<Mutex<CanvasCache>>
}

impl Context {
	pub fn new(name:&str, 
               use_weighting: bool,
               use_triangles: bool,
               use_circles: bool, 
               use_rectangles: bool) -> Context{
		let f = File::open(name).expect("failed to open file");
		let buf = BufReader::new(f);
		let mut jpg = Decoder::new(buf);
		let image = jpg.decode().expect("failed to decode image");
		let meta = jpg.info().unwrap();
        let depth = match meta.pixel_format {
            jpeg_decoder::PixelFormat::RGB24 => 3,
            jpeg_decoder::PixelFormat::L8 => 1,
            jpeg_decoder::PixelFormat::CMYK32 => 4
        };

		return Context {
			image: Canvas::from(meta.width as usize, meta.height as usize, depth, image),
            weightings: Canvas::new(meta.width as usize, meta.height as usize, 3),
			height: meta.height as i32,
			width: meta.width as i32,
            depth: depth as i32,
            format: meta.pixel_format,
            mutations: 0,
            use_weighting: use_weighting,
            use_triangles: use_triangles,
            use_circles: use_circles,
            use_rectangles: use_rectangles,
            cache: Arc::new(Mutex::new(CanvasCache::new(meta.width as usize, meta.height as usize, depth)))
		};
	}

    pub fn weight_entropy(&mut self) {
        // Copy image
        for x in 0..self.width {
            for y in 0..self.height {
                let diff = ((self.image.neighbors_diffsq(x, y, 1) / 10) as f32).sqrt().min(255.);
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

