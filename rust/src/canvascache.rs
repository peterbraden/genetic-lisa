extern crate chrono;
extern crate lru_cache;

use canvas::{Canvas};
use self::lru_cache::LruCache;
use shapelist::{ShapeList};
use shapes::{Shape};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Debug)]
pub struct CanvasCache {
    map: LruCache<u64, Canvas>,
    width: usize,
    height: usize,
    depth: usize,
    hits: usize,
    misses: usize,
    requests: usize,
    shapes: usize
}

fn calculate_hash(t: &[Shape]) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    return s.finish();
}

impl CanvasCache {

    pub fn new(width: usize, height: usize, depth: usize) -> CanvasCache {
        CanvasCache {
            map: LruCache::new(1000),
            width: width,
            height: height,
            depth: depth,
            hits: 0,
            misses: 0,
            requests: 0,
            shapes: 0
        }
    }   

    fn fallback_for(sl: &ShapeList, width:usize, height:usize, depth:usize) -> Canvas {
        let mut canv = Canvas::new(width, height, depth);
        sl.draw_onto(&mut canv);
        return canv;
    }

    pub fn canvas_for(&mut self, sl: &ShapeList) -> Canvas {
        self.requests += 1;
        if (self.requests) % 1000 == 0 {
            let now = chrono::Utc::now();
            print!("{} Cache: hits:{} misses: {} req: {} shp: {}  len: {}\n", 
                  now, self.hits, self.misses, self.requests, self.shapes, self.map.len());
        }
        return self.search_sublist(sl);
    }

    /// Insert a shapelist and all subportions of that shapelist
    pub fn insert(&mut self, sl: &ShapeList) {
        let mut canv = Canvas::new(self.width, self.height, self.depth);
        for i in 0..sl.len() {
            let s = sl.slice(i + 1);
            sl.draw_item_onto(i, &mut canv);
            self.map.insert(calculate_hash(&s.shapes), canv.clone()); 
        }
    }


    /// Because a shapelist is an ordered series of mutations to a canvas
    /// and because most of the time it will have a common initial portion
    /// of the list with a cached canvas, we can iterate backwards through
    /// the list until we find a cached portion, and then draw onto a clone
    /// from there.
    ///
    /// This means that mutations to a shapelist are cheaper towards the end
    pub fn search_sublist(&mut self, sl: &ShapeList) -> Canvas {
        for i in 0..sl.len() {
            let h = calculate_hash(&sl.shapes[0..i]);
            match self.map.get_mut(&h){
                Some(k) => { 
                    self.hits += 1;
                    self.shapes += i;
                    let mut c = k.clone();
                    for x in (sl.len() - i)..sl.len() {
                        sl.draw_item_onto(x, &mut c);
                    }
                    return c;
                },
                None => {}
            }
        }
        
        // None found, 
        self.misses += 1;
        return CanvasCache::fallback_for(&sl, self.width, self.height, self.depth);
    }
}
