use shapes::{Shape};
use std::fmt::Write;
use canvas::{Canvas};
use rando::{rand, choose};
use context::Context;

#[derive(PartialEq, Eq, Hash, Debug, Clone, Serialize, Deserialize)]
pub struct ShapeList {
    pub shapes: Vec<Shape>
}

impl ShapeList {
    pub fn new() -> ShapeList {
        ShapeList {
            shapes: Vec::new()
        }
    }

    pub fn slice(&self, i: usize) -> ShapeList{
        let n = (&self.shapes[0..i]).to_vec();
        return ShapeList { shapes: n }
    }

    fn remove_random(&mut self) {
        let i = (rand() * self.shapes.len() as f32) as usize;
        self.shapes.remove(i);
    }

    pub fn add_random(&mut self, ctx: &Context) { 
        self.shapes.push(Shape::random(ctx.use_triangles, ctx.use_circles, ctx.use_rectangles));
    }

    pub fn remove_shape(&mut self) {
        if self.shapes.len() > 1 {
            self.remove_random();
        }
    }
    
    pub fn swap(&mut self){
        if self.len() > 2 {
            let i = (rand() * self.shapes.len() as f32) as usize;
            let i2 = (rand() * self.shapes.len() as f32) as usize;
            if i != i2 {
                self.shapes.swap(i, i2);
            }
        }
    }

    pub fn mutate(&mut self) {
        match (rand() * 100.) as u8 {
            0...90 => {
                // Bias end mutations as they're cheaper
                let l = self.shapes.len();
                if l > 0 {
                    self.shapes[l - 1].mutate();
                }
            },
            90...100 => {
                match choose(&mut self.shapes) {
                    Some(m) => { m.mutate(); }
                    None => {}
                }
            },
            _ => panic!()
        }
    }


    pub fn len(&self) -> usize {
        return self.shapes.len();
    }

	pub fn svg(&self, width: usize, height: usize, depth: usize) -> String {
		let mut out = String::new();
		let mut contents = String::new();
		for c in &self.shapes{
			contents.push_str(&c.svg(width, height, depth));
		}
        let svgprelude = "svg xmlns='http://www.w3.org/2000/svg' style='background-color: #000;' ";
		write!(&mut out, "<{} width='{}' height='{}' >{}</svg>",
                svgprelude, width, height, contents)
                .expect("String concat failed");
		return out;
	}

    pub fn draw_onto(&self, mut canv: &mut Canvas) {
		for c in &self.shapes{
            c.draw_onto(&mut canv);
		}
    }

    pub fn draw_item_onto(&self, i: usize, canv: &mut Canvas) {
        self.shapes[i].draw_onto(canv);
    }

    pub fn to_string(&self) -> String {
		let mut out = String::new();
		for c in &self.shapes{
            out.push_str(&c.to_string());
        }
        return out;
    }
}
