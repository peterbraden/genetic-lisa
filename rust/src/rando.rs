extern crate rand as _rand;
use rando::_rand::Rng;

pub fn rand() -> f32 {
	return _rand::thread_rng().gen_range(0.,1.);
}

pub fn randu8() -> u8 {
	return _rand::thread_rng().gen_range(0,255);
}

pub fn randu8f() -> f32 {
	return _rand::thread_rng().gen_range(0,255) as f32;
}

pub fn rand_color_adjust(c:f32, range: f32) -> f32 {
	return (c + ((rand() - 0.5) * 256.0 * range)).min(255.).max(0.);
}

pub fn rand_adjust(p:f32, range: f32, min: f32, max:f32) -> f32 {
    return (p + ((rand() - 0.5) * range)).min(max).max(min);
}

pub fn choose<T>(v: &mut Vec<T>) -> Option<&mut T> {
    return _rand::thread_rng().choose_mut::<T>(v);
}
