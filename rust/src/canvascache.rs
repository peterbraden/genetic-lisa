extern crate chrono;
extern crate lru_cache;

use canvas::{Canvas};
use self::lru_cache::LruCache;
use shapelist::{ShapeList};

#[derive(Debug)]
pub struct CanvasCache {
    map: LruCache<String, Canvas>,
    width: usize,
    height: usize,
    depth: usize,
    hits: usize,
    misses: usize,
    requests: usize
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
            requests: 0
        }
    }   

    fn fallback_for(sl: &ShapeList, width:usize, height:usize, depth:usize) -> Canvas {
        let mut canv = Canvas::new(width, height, depth);
        sl.draw_onto(&mut canv);
        return canv;
    }

    
    fn get_or_insert(mut c: &mut LruCache<String, Canvas>, sl: &ShapeList, width:usize, height:usize, depth:usize) -> Canvas {
        let k = sl.to_string();

        if !c.contains_key(&k) {
            c.insert(k.clone(), CanvasCache::fallback_for(
                     &sl,  width, height, depth));
        }

        return c.get_mut(&k).unwrap().clone();
    }

    pub fn canvas_for(&mut self, sl: &ShapeList) -> Canvas {
        self.requests += 1;
        if (self.requests) % 100 == 0 {
            let now = chrono::Utc::now();
            print!("{} Cache: hits:{} misses: {} req: {} len: {}\n", 
                  now, self.hits, self.misses, self.requests, self.map.len());
        }
        return self.search_sublist(sl, 0);
    }

    /// Insert a shapelist and all subportions of that shapelist
    pub fn insert(&mut self, sl: &ShapeList) {
        for i in 0..sl.len() {
            CanvasCache::get_or_insert(
                &mut self.map, &sl.slice(sl.len() - i),
                self.width, self.height, self.depth);
        }
    }


    /// Because a shapelist is an ordered series of mutations to a canvas
    /// and because most of the time it will have a common initial portion
    /// of the list with a cached canvas, we can iterate backwards through
    /// the list until we find a cached portion, and then draw onto a clone
    /// from there.
    ///
    /// This means that mutations to a shapelist are cheaper towards the end
    pub fn search_sublist(&mut self, sl: &ShapeList, i: usize) ->  Canvas {
        let ind = sl.len() - i;
        let k = sl.slice(ind).to_string();

        if ind <= 1 {
            // Matching the first shape, so just return or add to the cache
            self.misses += 1;
            return CanvasCache::get_or_insert(
                &mut self.map, &sl.slice(ind), self.width, self.height, self.depth).clone()
        }

        if self.map.contains_key(&k) {
            self.hits += 1;
            return self.map.get_mut(&k).unwrap().clone();
        }

        // Get the canvas of the sublist before and add this shape
        let mut c = self.search_sublist(sl, i + 1); 
        sl.draw_item_onto(ind - 1, &mut c);
        return c;
        //return self.map.entry(k).or_insert(c).clone(); // CACHE ALL
    }
}
