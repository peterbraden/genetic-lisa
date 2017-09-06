extern crate chrono;
extern crate lru_cache;

use canvas::{Canvas};
use self::lru_cache::LruCache;
use shapelist::{ShapeList};

#[derive(Debug)]
pub struct CanvasCache {
    map: LruCache<ShapeList, Canvas>,
    width: usize,
    height: usize,
    depth: usize,
    hits: usize,
    misses: usize,
    requests: usize,
    shapes: usize
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

    
    fn get_or_insert(mut c: &mut LruCache<ShapeList, Canvas>, sl: &ShapeList, width:usize, height:usize, depth:usize) -> Canvas {
        if !c.contains_key(&sl) {
            c.insert(sl.clone(), CanvasCache::fallback_for(
                     &sl,  width, height, depth));
        }

        return c.get_mut(&sl).unwrap().clone();
    }

    pub fn canvas_for(&mut self, sl: &ShapeList) -> Canvas {
        self.requests += 1;
        if (self.requests) % 1000 == 0 {
            let now = chrono::Utc::now();
            print!("{} Cache: hits:{} misses: {} req: {} shp: {}  len: {}\n", 
                  now, self.hits, self.misses, self.requests, self.shapes, self.map.len());
        }
        return self.search_sublist_non_recursive(sl);
    }

    /// Insert a shapelist and all subportions of that shapelist
    pub fn insert(&mut self, sl: &ShapeList) {
        let mut canv = Canvas::new(self.width, self.height, self.depth);
        for i in 0..sl.len() {
            let s = sl.slice(i + 1);
            sl.draw_item_onto(i, &mut canv);
            self.map.insert(s, canv.clone()); 
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
        let k = sl.slice(ind);

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

    pub fn search_sublist_non_recursive(&mut self, sl: &ShapeList) -> Canvas {
        for i in 0..sl.len() {
            match self.map.get_mut(&sl.slice(sl.len() - i)){
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
