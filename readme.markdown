The Evolution of a Smile
========================

A collaborative evolutionary algorithm that aims to reproduce the Mona Lisa with overlapping translucent circles.



Introduction
------------

In December 2008, [Roger Alsing](http://rogeralsing.com/2008/12/07/genetic-programming-evolution-of-mona-lisa/) wrote a program that used a genetic algorithm to evolve an approximation of the mona lisa using 50 overlapping polygons. The results were very impressive and his report generated a lot of interest.

I was curious to try it myself, and wrote a quick version in python. At the time, the source for Roger's program wasn't available (he's subsequently released a .NET program with source, although I haven't looked at it), so I guessed at how the program worked.

Instead of using a fixed set of polygons, I decided to use ellipses, initially because the maths was easier, but subsequently because I like the abstract 'lava lamp' look of the intermediary images. Rather than use a fixed array of shapes like Roger, I used a fitness function that encouraged less shapes, and allowed my mutation operation to add or remove ellipses.

Despite experimenting with a bunch of different image libraries, and using psyco, the python program was prohibitively slow, only evolving a few thousand generations per hour. I experimented with generation populations, varying between 10 and 100 - it was interesting looking at how adjusting the parameters affected the speed at which the fitness improved.

![One of the python versions](http://github.com/peterbraden/genetic-lisa/raw/master/lisa/old-versions/mutation-64225.jpg)


As the bottleneck in the program seemed to be the graphics library, I decided to rewrite the program in c, using arrays of pixels to manipulate so the fitness function would be blazing fast. Despite my rusty C skills, the many memory leaks, and platform inconsistencies (OS X zero fills malloc allocations, linux does not), I got a version running and left it overnight on a friends server.

Using a generation population of 100, in 19000 generations I had something that looked vaguely Da Vinciesque. If you squinted, it looked pretty good. Curiosity satisfied, I abandoned the project.

![The result from the C Program](http://github.com/peterbraden/genetic-lisa/raw/master/lisa/old-versions/19690.png)

A year later and I'm playing with html5. I'm interested in splitting computation between multiple browsers. One of my projects, a collaborative map-reduce raytracer, uses multiple browsers to render an image into canvas. I'm excited by the capabilities of canvas, and start writing more and more javascript. My server side code, a mess of python and mysql, limits the interest I have in doing anything further with this.

A few months ago I started toying with the idea of using a couchapp to simplify the collaborative processing backend. Looking through my old projects, the mona lisa code seems perfect for this, so I rewrite the genetic algorithm in javascript with the canvas.

This time I'm using circles instead of ellipses. And I add the ability to cross breed between browsers. In 10,000 generations or so, I'm getting interesting images. And this time, you can see the images evolving. 


Check it out
------------

Currently, I haven't been able to get my couchone database to be public, although this should be done in the next few days.
