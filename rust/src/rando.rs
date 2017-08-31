extern crate rand as _rand;
use rando::_rand::Rng;

pub fn rand() -> f64 {
	return _rand::thread_rng().gen_range(0.,1.);
}

pub fn randu8() -> u8 {
	return _rand::thread_rng().gen_range(0,255);
}

pub fn rand_color_adjust(c:u8) -> u8 {
	return c.saturating_add(((rand() - 0.5) * 256.0) as u8);
}

pub fn rand_adjust(p:f64, range: f64, min: f64, max:f64) -> f64 {
    return (p + ((rand() - 0.5) * range)).min(max).max(min);
}

pub fn choose<T>(v: &mut Vec<T>) -> Option<&mut T> {
    return _rand::thread_rng().choose_mut::<T>(v);
}
