#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import <math.h>

#define POPULATION 100
#define DIM 250*389
#define DIMX 250
#define DIMY 389
#define COORD(x,y) ((y)*250 + (x))
#define MIN(x,y) (x>y?y:x)

typedef struct circ{
	int x, y, rad, color;
	float opacity;
	struct circ* next;
} circle;

int* loadTarget(){
	void* fp = fopen("mldata", "r");
	int i =0;
	int x;
	int* data = malloc(DIM * sizeof(int));
	while (!feof(fp)) {
		if (fscanf(fp, "%d\n", data + i)!=1){
			break;
		} 
		i++;
	}
	return data;
}


/**
 * Convert an array of ints to a comma seperated string
 */
char* intsToChars(int* arr, int dim){
	int i;
	int ind = 1;
	
	char *out = malloc(dim * 2 * (sizeof(int) + 1) * sizeof(char));

	for(i=0; i<dim; ++i){
		ind += sprintf(out+ind-1, "%d,", arr[i]);
	}
	out[ind -2] = '\0';
	return out;	
}

/*
 * A fitness function to compare between 2 images.
 */
int fitness(int* im, int objcount, int* targ, int size){
	unsigned int diff = 0;
	int i;
	int r,g,b;
	for (i=0; i<size; ++i){
		r = ((targ[i]>>16) % 256) - ((im[i]>>16) % 256);
		g = ((targ[i]>>8) % 256) - ((im[i]>>8) % 256);
		b = (targ[i] % 256) - (im[i] % 256);
		diff += r*r + g*g + b*b;
	}
	return diff; // * (1+objcount/100);
}

/*
 * Create Image
 */
int* allocateImage(){
	int* im = malloc(DIM * sizeof(int));
	int i;
	for (i=0; i<DIM; i++){
		im[i]=0;
	}
	return im;
}


/**
 * Overlay color
 */
int resolveColor(int existing, int color, float opacity){
	int r = MIN(((existing >> 16) % 256 * (1.0-opacity)) + (((color >> 16) % 256) * opacity),255);
	int g = MIN(((existing >> 8) % 256 * (1.0-opacity)) +(((color >> 8) % 256) * opacity), 255);
	int b = MIN((existing % 256 * (1.0-opacity)) + ((color % 256) * opacity), 255);

	return (r << 16) + (g << 8) + b;
}

/*
 * Draw a circle upon the image
 */
void drawCircle(int* im, circle* c){
	int x, height, y;
	for (x = -c->rad; x <= c->rad; x++){
    	height = (int) sqrt(c->rad * c->rad - x * x);
	    for (y = -height; y <= height; y++){
	        if ((x+c->x) >0 && (y + c->y>0) && COORD(x + c->x, y + c->y) < DIM){
	        	im[COORD(x + c->x, y + c->y)]= resolveColor(im[COORD(x + c->x, y + c->y)], c->color, c->opacity);
	    	}
	    }
	}
}

void redraw(int* im, circle* c){
	int i;
	for (i=0;i<DIM; i++){
		im[i] = 0;	
	}

	if (c!=NULL){
		circle* f;
		for(f=c; f!=NULL; f = f->next){
			drawCircle(im, f);
		}
	}	
}

int countCircles(circle* e){
	if (e == NULL)
		return 0;
	return countCircles(e->next) + 1;
}

int rnd(int max){
	return rand() % max + 1;
}

circle* mutate(int* im, circle* c){
	if (rnd(2) > 1){
		/* Add Random Circle */
		circle* d = (circle *) malloc(sizeof(circle));
		d->x = rnd(DIMX);
		d->y = rnd(DIMY);
		d->color = rnd(0xffffff);
		d->rad = rnd(100);
		d->opacity = ((float) (rnd(10000))/30000) + 0.1;
		d->next = NULL;

		// Put on end so doesn't screw prev image
		if (c!=NULL){
			circle* f;
			circle* p;
			for(f=c;f!=NULL;f = f->next){
				p=f;
			}
			p->next = d;
		}else{
			c = d;
			c->next = NULL;
		}
		drawCircle(im, d);

	}else{
		/* Delete Random Circle */
		circle* f = c;
		int num = countCircles(c)-1;
		
		if (num >1){
			int ind = rand() % num;
			circle* old;
			
			if (ind==0){
				old = c;
				c = c->next;
				
			}else{
				
				int i;
				for (i=0; i<ind; i++){
					f = f->next;
				}
				
				old = (circle*) f->next;
				f->next = f->next->next;
			}
			old->next =NULL;
			free(old);	
			redraw(im, c);
				
		}else{
			if(c!=NULL){
				free(c);
			}	
			c = NULL;
		}		
	}
	if (rnd(2)>1)
		c = mutate(im, c);
	return c;
}

void freeCircles(circle* e){
	if (e != NULL){
		if (e->next){
			freeCircles(e->next);
		}
		e->next = NULL;
		free(e);
	}
}

void cloneImage(int* old, int* nw){
	memcpy(old, nw, DIM*sizeof(int));
}

		


circle* cloneCircles(circle* e){
	if (e == NULL){
		return NULL;
	}	
	
	circle* d = (circle *) malloc(sizeof(circle));
	d->x = e->x;
	d->y = e->y;
	d->color = e->color;
	d->rad = e->rad;
	d->opacity =e->opacity;
	
	
	if (e->next != NULL){
		d->next = cloneCircles(e->next);
	}else{
		d->next = NULL;
	}
	return d;
}

void printCircles(circle* e, FILE* fl){
	if (e==NULL){
		return;}
	fprintf(fl, "<%d,%d |%d : %x %f >", e->x, e->y, e->rad, e->color, e->opacity);
	printCircles(e->next, fl);
}


int main(char* args){
	srand(1);
	int* dat = loadTarget();
	int i;
	
	int** images = malloc(POPULATION*sizeof(int*));
	circle** data = malloc(POPULATION*sizeof(circle*));
	
	for(i=0;i<POPULATION; i++){
		images[i] = allocateImage();
		data[i] = NULL;
	}	
	 
	 
	unsigned int min = 0xffffffff;
	unsigned int prev = min;
	circle* best = NULL;
	int min_ind = 0;
	int itrs =0;
	
	while (1){
		for(i =0; i<POPULATION; i++){
			data[i] = mutate(images[i], data[i]);
			int f = fitness(images[i], countCircles(data[i]), dat, DIM);
			//printCircles(data[i]);
			//printf("\t\tf: %d\n", fitness(images[i], countCircles(data[i]), dat, DIM));
			
			if (f<min){
				min = f;
				min_ind = i;
				best = cloneCircles(data[i]);
			}		
		}
		
		if (min<prev){
			printf("new best: %d - %d circles\n", min, countCircles(data[min_ind]));
			prev = min;
			
			FILE* fp = fopen("best", "w");
			fprintf(fp, "%s", intsToChars(images[min_ind], DIM));
			fclose(fp);
			
			FILE* fp2 = fopen("bestcirc", "a");
			fprintf(fp2, "\n\n\nGEN %d\n\n\n", itrs);
			printCircles(best, fp2);
			fclose(fp2);
		}
		
		if(itrs%10==0){
			//printf("\n\n\n");
			//printCircles(best);
			printf("\t->%d, %d \n" , itrs, min);	
		}
		
		for(i =0; i<POPULATION; i++){
			if (i != min_ind){
				//Clone circles
				freeCircles(data[i]);
				data[i] = cloneCircles(best);
				//Clone images
				cloneImage(images[i], images[min_ind]); 
			}
		}	

		itrs ++;
	}
	return 0;
}
