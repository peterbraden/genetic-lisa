use canvas::{Canvas};
use rando::{rand, rand_adjust, randu8};
use color::Color;
use std::fmt::Write;
use std::cmp::{min, max};
use std::hash::{Hash, Hasher};

pub trait ShapeBehaviour {
    fn mutate(&mut self);
    fn svg(&self, width: usize, height: usize) -> String;
    fn to_string(&self) -> String;
    fn draw_onto(&self, &mut Canvas);
}

#[derive(PartialEq, Hash, Eq, Debug, Clone, Serialize, Deserialize)]
pub enum Shape {
    Circle(Circle),
    Rect(Rect),
    Triangle(Triangle)
}

impl Shape {
    pub fn random() -> Shape {
        match (rand() * 10.) as u8 {
            0...4 => { return Shape::Circle(Circle::random()) },
            4...7 => { return Shape::Triangle(Triangle::random()) },
            7...10 => { return Shape::Rect(Rect::random()) },
            _ => panic!("Unknown shape")
        }
    
    }

    pub fn mutate(&mut self) {
        match self {
            &mut Shape::Triangle(ref mut t) => t.mutate(),
            &mut Shape::Rect(ref mut r) => r.mutate(),
            &mut Shape::Circle(ref mut c) => c.mutate()
        }
    }

    pub fn svg(&self, width: usize, height: usize) -> String{
        match self {
            &Shape::Triangle(ref t) => t.svg(width, height),
            &Shape::Rect(ref t) => t.svg(width, height),
            &Shape::Circle(ref c) => c.svg(width, height)
        }
    }

    pub fn to_string(&self) -> String{
        match self {
            &Shape::Triangle(ref t) => t.to_string(),
            &Shape::Rect(ref t) => t.to_string(),
            &Shape::Circle(ref c) => c.to_string()
        }
    }

    #[inline]
    pub fn draw_onto(&self, mut canv: &mut Canvas) {
        match self {
            &Shape::Triangle(ref t) => t.draw_onto(canv),
            &Shape::Rect(ref t) => t.draw_onto(canv),
            &Shape::Circle(ref c) => c.draw_onto(canv)
        }
    }
}

#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub color: Color
}

impl Rect {
    pub fn random() -> Rect {
        Rect {
            x: rand(),
            y: rand(),
            width: rand(),
            height: rand(),
            color: Color {
                r: randu8(),
                g: randu8(),
                b: randu8(),
                opacity: rand()
            }
        }
    }

}

impl ShapeBehaviour for Rect {
    fn mutate(&mut self) {
        match (rand() * 100.) as u8 {
            0...60 => self.color = self.color.mutate(),
            60...70 => self.x += rand_adjust(self.x, 0.5, 0., 1.0),
            70...80 => self.y += rand_adjust(self.y, 0.5, 0., 1.0),
            80...90 => self.width += rand_adjust(self.width, 0.5, 0., 1.0),
            90...100 => self.height += rand_adjust(self.height, 0.5, 0., 1.0),
            _ => panic!()
        }
    }

    fn to_string(&self) -> String {
        return format!("<R{:.6},{:.6},{:.6},{:.6},{}>", self.x, self.y, self.width, self.height, self.color.rgba());
    }

    fn svg(&self, width: usize, height: usize) -> String {
		let mut out = String::new();
		write!(&mut out, "<rect x='{}' y='{}' width='{}' height='{}' fill='{}' />",
                (self.x * width as f32) as i32,
                (self.y * height as f32) as i32,
                ((self.width) * width as f32) as i32,
                ((self.height) * height as f32) as i32,
                self.color.rgba())
			.expect("String concat failed");
		return out;
    }

    #[inline]
    fn draw_onto(&self, canv: &mut Canvas) {
        let x1 = (self.x * canv.width as f32) as i32;
        let y1 = (self.y * canv.height as f32) as i32;
        let x2 = x1 + (self.width * canv.width as f32) as i32;
        let y2 = y1 + (self.height * canv.height as f32) as i32;
        let xmin = max(x1, 0);
        let xmax = min(x2, canv.width as i32);
        let ymin = max(y1, 0);
        let ymax = min(y2, canv.height as i32);
        for x in  xmin .. xmax {
            for y in ymin .. ymax {
                canv.add_pixel(x, y, &self.color)
            }
        }
    }
}

impl Hash for Rect {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let x = (self.x * 1000.) as i32;
        let y = (self.y * 1000.) as i32;
        let width = (self.width * 1000.) as i32;
        let height = (self.height * 1000.) as i32;
        x.hash(state);
        y.hash(state);
        width.hash(state);
        height.hash(state);
        self.color.hash(state);
    }
}

impl Eq for Rect {}

#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct Triangle {
    pub x1: f32,
    pub x2: f32,
    pub x3: f32,
    pub y1: f32,
    pub y2: f32,
    pub y3: f32,
    pub color: Color
}

impl Triangle {
    pub fn random() -> Triangle {
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
}

impl ShapeBehaviour for Triangle {
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

    fn to_string(&self) -> String {
        return format!("<T{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}>", self.x1, self.y1,
                                                        self.x2, self.y2, 
                                                        self.x3, self.y3, 
                                                        self.color.rgba());
    }

    fn svg(&self, width: usize, height: usize) -> String {
		let mut out = String::new();
		write!(&mut out, "<polygon points='{},{} {},{} {},{}' fill='{}' />",
                (self.x1 * width as f32) as i32,
                (self.y1 * height as f32) as i32,
                (self.x2 * width as f32) as i32,
                (self.y2 * height as f32) as i32,
                (self.x3 * width as f32) as i32,
                (self.y3 * height as f32) as i32,
                self.color.rgba())
			.expect("String concat failed");
		return out;
    }

    #[inline]
    fn draw_onto(&self, canv: &mut Canvas) {
        let x1 = (self.x1 * canv.width as f32) as i32;
        let y1 = (self.y1 * canv.height as f32) as i32;
        let x2 = (self.x2 * canv.width as f32) as i32;
        let y2 = (self.y2 * canv.height as f32) as i32;
        let x3 = (self.x3 * canv.width as f32) as i32;
        let y3 = (self.y3 * canv.height as f32) as i32;
        let xmin = min(x1, min(x2, x3));
        let xmax = min(max(x1, max(x2, x3)), canv.width as i32);
        let ymin = min(y1, min(y2, y3));
        let ymax = min(max(y1, max(y2, y3)), canv.height as i32);
        
        for x in xmin .. xmax  {
            for y in ymin .. ymax {
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

impl Hash for Triangle {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let x1 = (self.x1 * 1000.) as i32;
        let y1 = (self.y1 * 1000.) as i32;
        let x2 = (self.x2 * 1000.) as i32;
        let y2 = (self.y2 * 1000.) as i32;
        let x3 = (self.x3 * 1000.) as i32;
        let y3 = (self.y3 * 1000.) as i32;
        x1.hash(state);
        y1.hash(state);
        x2.hash(state);
        y2.hash(state);
        x3.hash(state);
        y3.hash(state);
        self.color.hash(state);
    }
}

impl Eq for Triangle {}

#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct Circle {
	pub x: f32,
	pub y: f32,
	pub rad: f32,
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

    /*
    pub fn merge(&mut self, d: Circle) {
        self.color = (&self.color + &d.color) * 0.5;
        self.x = (self.x + d.x) / 2.;
        self.y = (self.y + d.y) / 2.;
        self.rad = (self.rad + d.rad) / 2.;
    }
    */

    #[inline]
    pub fn draw_onto_slow(&self, mut canvas: &mut Canvas) {
        let rad = (self.rad * canvas.width as f32) as i32;
        let cx = (self.x * canvas.width as f32) as i32;
        let cy = (self.y * canvas.height as f32) as i32;
		let radrad = rad * rad;

		for x in -rad .. rad {
			for y in -rad .. rad {
                if x*x + y*y < radrad {
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

impl ShapeBehaviour for Circle {
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
		let cx = (self.x * width as f32) as i32;
		let cy = (self.y * height as f32) as i32;
		let rad = (self.rad * width as f32) as i32;
		write!(&mut out, "<circle cx='{}' cy='{}' r='{}' fill='{}' />",
                cx, cy, rad, self.color.rgba())
			.expect("String concat failed");
		return out;
	}
    fn to_string(&self) -> String {
        return format!("<C{:.6},{:.6},{:6},{}>", self.x, self.y, self.rad, self.color.rgba());
    }

    #[inline]
    fn draw_onto(&self, mut canvas: &mut Canvas) {
        // Bresenheim
        let rad = (self.rad * canvas.width as f32) as i32;
        let mut x = rad - 1;
        let mut y = 0;
		let cx = (self.x * canvas.width as f32) as i32;
		let cy = (self.y * canvas.height as f32) as i32;
        let mut dx = 1;
        let mut dy = 1;
        let mut err = dx - (rad << 1);

        while x >= y {
            canvas.line_add(cx - x, cx + x, cy + y, &self.color);
            canvas.line_add(cx - x, cx + x, cy - y, &self.color);
            canvas.line_add(cx - y, cx + y, cy + x, &self.color);
            canvas.line_add(cx - y, cx + y, cy - x, &self.color);

            if err <= 0 {
                y += 1;
                err += dy;
                dy += 2;
            }

            if err > 0 {
                x -= 1;
                dx += 2;
                err += (-rad << 1) + dx;
            }
        }

    }

}

impl Hash for Circle{
    fn hash<H: Hasher>(&self, state: &mut H) {
        let x = (self.x * 1000.) as i32;
        let y = (self.y * 1000.) as i32;
        let rad = (self.rad * 1000.) as i32;
        x.hash(state);
        y.hash(state);
        rad.hash(state);
        self.color.hash(state);
    }
}

impl Eq for Circle {}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn draw_shapes(){
        let mut c = Canvas::new(200, 200, 3);
        let mut c2 = Canvas::new(200, 200, 3);
        let s = Circle {
			x: 0.5,
			y: 0.5,
			rad: 0.5,
            color: Color {
                r: 100,
                g: 200,
                b: 250,
                opacity: 1.
            }
		};//Circle::random();
        s.draw_onto(&mut c);
        s.draw_onto_slow(&mut c2);
        //c.save("test.png");
        //c2.save("test2.png");
        //assert_eq!(c.pixels, c2.pixels);
    }
}
