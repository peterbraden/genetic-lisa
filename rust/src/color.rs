use std::fmt::Write;
use std::fmt;
use std::ops::{Mul, Add};
use rando::{rand, rand_adjust, rand_color_adjust};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub opacity: f64
}

impl Color {
    pub fn rgba(&self) -> String {
		let mut rgb = String::new();
        write!(&mut rgb,
               "rgba({},{},{},{:.4})",
                self.r, self.g, self.b, self.opacity)
            .expect("String concat failed");
        return rgb;
    }

    pub fn mutate(&self) -> Color{
        match (rand() * 100.) as u8 {
            0...25 => { return Color{
                                r: rand_color_adjust(self.r),
                                g: self.g,
                                b: self.b,
                                opacity: self.opacity,
                            }
                        },
            26...50 => { return Color{
                                r: self.r,
                                g: rand_color_adjust(self.g),
                                b: self.b,
                                opacity: self.opacity,
                            }
                        },
            51...75 => { return Color{
                                r: self.r,
                                g: self.g,
                                b: rand_color_adjust(self.b),
                                opacity: self.opacity,
                            }
                        },
            76...100 =>{ return Color{
                                r: self.r,
                                g: self.g,
                                b: self.b,
                                opacity: rand_adjust(self.opacity, 0.1, 0., 1.),
                            }
                        },
            _ => panic!()
        }
    }

}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    return write!(f,
            "rgba({},{},{},{:.4})",
            self.r, self.g, self.b, self.opacity)
    }
}

impl <'a, 'b> Add<&'b Color> for &'a Color {
    type Output = Color;

    fn add(self, _rhs: &Color) -> Color {
        Color {
            r: self.r + _rhs.r,
            g: self.g + _rhs.g,
            b: self.b + _rhs.b,
            opacity: self.opacity + _rhs.opacity
        }
    }
}

impl Mul<f64> for Color {
    type Output = Color;

    fn mul(self, _rhs: f64) -> Color {
        Color {
            r: (self.r as f64 * _rhs) as u8,
            g: (self.g as f64 * _rhs) as u8,
            b: (self.b as f64 * _rhs) as u8,
            opacity: self.opacity * _rhs
        }
    }
}
