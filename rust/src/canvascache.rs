extern crate chrono;
extern crate lru_cache;
extern crate fnv;

use canvas::{Canvas};
use self::lru_cache::LruCache;
use shapelist::{ShapeList};
use shapes::Shape;
use std::hash::{BuildHasherDefault};
use self::fnv::FnvHasher;

#[derive(Debug)]
pub struct CanvasCache {
    map: LruCache<Vec<Shape>, Canvas, BuildHasherDefault<FnvHasher>>,
    width: usize,
    height: usize,
    depth: usize,
    hits: usize,
    misses: usize,
    requests: usize,
    shapes: usize
}

fn fallback_for(sl: &ShapeList, width:usize, height:usize, depth:usize) -> Canvas {
    let mut canv = Canvas::new(width, height, depth);
    sl.draw_onto(&mut canv);
    return canv;
}

impl CanvasCache {

    pub fn new(width: usize, height: usize, depth: usize) -> CanvasCache {
        CanvasCache {
            map: LruCache::with_hasher(
                     1000,
                     BuildHasherDefault::<FnvHasher>::default()),
            width: width,
            height: height,
            depth: depth,
            hits: 0,
            misses: 0,
            requests: 0,
            shapes: 0
        }
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
            self.map.insert(s.shapes.clone(), canv.clone()); 
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
            let k = &sl.slice(sl.len() - i).shapes;
            match self.map.get_mut(k){
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
        return fallback_for(&sl, self.width, self.height, self.depth);
    }
}
