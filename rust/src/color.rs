use std::fmt::Write;
use std::fmt;
use std::ops::{Mul, Add};
use rando::{rand, rand_adjust, rand_color_adjust};
use std::hash::{Hash, Hasher};

#[inline]
pub fn color_add(c:f32, c2: f32, opacity: f32) -> f32 {
	return c * (1. - opacity) + (c2 * opacity);
}

#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct Color {
    pub r: f32, // 0 - 255 (u8)
    pub g: f32,
    pub b: f32,
    pub opacity: f32
}

impl Color {
    pub fn svg(&self, depth: usize) -> String {
        if depth == 1 {
            let mut out = String::new();
            write!(&mut out,
                   "rgba({},{},{},{:.4})",
                    self.r as u8, self.r as u8, self.r as u8, self.opacity)
                .expect("String concat failed");
            return out;
        }
        return self.rgba()
    }


    pub fn rgba(&self) -> String {
		let mut rgb = String::new();
        write!(&mut rgb,
               "rgba({},{},{},{:.4})",
                self.r as u8, self.g as u8, self.b as u8, self.opacity)
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

    #[inline]
    pub fn add_to_vec(&self, vec: &mut Vec<f32>, i: usize){
        vec[i]     = color_add(vec[i],      self.r, self.opacity);
        vec[i + 1] = color_add(vec[i + 1],  self.g, self.opacity);
        vec[i + 2] = color_add(vec[i + 2],  self.b, self.opacity);
    }

    pub fn black() -> Color {
        Color {r:0.,g:0.,b:0.,opacity:1.}
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    return write!(f,
            "rgba({},{},{},{:.4})",
            self.r as u8, self.g as u8, self.b as u8, self.opacity)
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

impl Mul<f32> for Color {
    type Output = Color;

    fn mul(self, _rhs: f32) -> Color {
        Color {
            r: self.r * _rhs,
            g: self.g * _rhs,
            b: self.b * _rhs,
            opacity: self.opacity * _rhs
        }
    }
}

impl Hash for Color {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.r as u8).hash(state);
        (self.g as u8).hash(state);
        (self.b as u8).hash(state);
        ((self.opacity * 1000.) as i32).hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_color_add() { 
        let c = color_add(255., 255., 1.);
        assert_eq!(c, 255);
    }
}
