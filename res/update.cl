#define INDEX(x,y)	((x) + SIZE_X * (y))

/*
conway's life rules:

	012345678
0	000100000
1	001100000

0 -> 3 of 1 -> 1
0 -> else -> 0

1 -> 2 of 1 -> 1
1 -> 3 of 1 -> 1
1 -> else -> 0

*/

__kernel void update(
	__global float4* dstBuffer,
	const __global float4* srcBuffer,
	__write_only image2d_t tex)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int2 i = (int2)(x, y);
	int index = INDEX(i.x, i.y);

	const __global float4* src = srcBuffer + index;
	__global float4* dst = dstBuffer + index;

	//count neighbors
	int xp = (i.x - 1 + SIZE_X) % SIZE_X;
	int xn = (i.x + 1) % SIZE_X;
	int yp = (i.y - 1 + SIZE_Y) % SIZE_Y;
	int yn = (i.y + 1) % SIZE_Y;

#if 1	//conway's life
	float neighbors = 
		srcBuffer[INDEX(xp, yp)].x +
		srcBuffer[INDEX(x, yp)].x +
		srcBuffer[INDEX(xn, yp)].x +
		srcBuffer[INDEX(xp, y)].x +
		srcBuffer[INDEX(xn, y)].x +
		srcBuffer[INDEX(xp, yn)].x +
		srcBuffer[INDEX(x, yn)].x +
		srcBuffer[INDEX(xn, yn)].x;

	float srci = srcBuffer[index].x;

	//if we are alive then keep alive on 2..3
	//if we are dead then grow on 3

	if (srci) {
		if (neighbors < 2.f || neighbors > 3.f) {
			dst->x = 0.f;
		} else {
			dst->x = 1.f;
		}
	} else {
		if (neighbors != 3.f) {
			dst->x = 0.f;
		} else {
			dst->x = 1.f;
		}
	}
	float4 color = (float4)(dst->x, 0.f, 0.f, 1.f);
	write_imagef(tex, i, color);
#endif

#if 0	//experimenting

	int neighbors[8] = {
		INDEX(xp, yp),
		INDEX(x, yp),
		INDEX(xn, yp),
		INDEX(xp, y),
		INDEX(xn, y),
		INDEX(xp, yn),
		INDEX(x, yn),
		INDEX(xn, yn)
	};

	//diffuse red
	float red = 0.f;
	for (int i = 0; i < 8; ++i) {
		red += srcBuffer[neighbors[i]].x;
	}
	red /= 8.f;

	//collect red
	//if we are the max of our neighbors then +1
	//else -1/8
	int maxIndex = 0;
	for (int i = 1; i < 8; ++i) {
		if (srcBuffer[neighbors[i]].x > srcBuffer[neighbors[maxIndex]].x) {
			maxIndex = i;
		}
	}
	if (red > srcBuffer[neighbors[maxIndex]].x) {
		red += 2.f;
	}

	//at low red values add blue
	float blue = src->z;
	if (red > .01f && red < .02f) {
		red -= 0.01f;
		blue = max(.7f, blue + .05f);
	} else if (red > .2f) {
		blue = max(0.f, blue - .02f);
	}

	
	float green = src->y;
	if (red > .2 && blue > .2f) {
		green += .05f;
	}
	if (red > 1.f) {
		green = max(0.f, green - .1f);
	}

	//punish overcrowding
	float nbhdGreen = 0.f;
	for (int i = 0; i < 8; ++i) {
		nbhdGreen += srcBuffer[neighbors[i]].y;
	}
	if (nbhdGreen < .2f || nbhdGreen > .5f) {
		green = max(0.f, green - 0.02f);
	} else if (nbhdGreen > .3f) {
		green += .02f;
	}


	dst->x = red;
	dst->y = green;
	dst->z = blue;
	dst->w = src->w;

	write_imagef(tex, i, *dst);

#endif
}

