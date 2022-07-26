/*
 * Copyright (c) 2011-2019, Peter Abeles. All Rights Reserved.
 *
 * This file is part of BoofCV (http://boofcv.org).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package boofcv.alg.transform.ii.impl;

import boofcv.alg.transform.ii.IntegralKernel;
import boofcv.struct.ImageRectangle;
import boofcv.struct.image.*;

import javax.annotation.Generated;


/**
 * <p>
 * Compute the integral image for different types of input images.
 * </p>
 * 
 * <p>
 * DO NOT MODIFY. This code was automatically generated by GenerateImplIntegralImageOps.
 * <p>
 * 
 * @author Peter Abeles
 */
@Generated("boofcv.alg.transform.ii.impl.GenerateImplIntegralImageOps")
public class ImplIntegralImageOps {

	public static void transform( final GrayF32 input , final GrayF32 transformed )
	{
		int indexSrc = input.startIndex;
		int indexDst = transformed.startIndex;
		int end = indexSrc + input.width;

		float total = 0;
		for( ; indexSrc < end; indexSrc++ ) {
			transformed.data[indexDst++] = total += input.data[indexSrc];
		}

		for( int y = 1; y < input.height; y++ ) {
			indexSrc = input.startIndex + input.stride*y;
			indexDst = transformed.startIndex + transformed.stride*y;
			int indexPrev = indexDst - transformed.stride;

			end = indexSrc + input.width;

			total = 0;
			for( ; indexSrc < end; indexSrc++ ) {
				total +=  input.data[indexSrc];
				transformed.data[indexDst++] = transformed.data[indexPrev++] + total;
			}
		}
	}

	public static void transform( final GrayF64 input , final GrayF64 transformed )
	{
		int indexSrc = input.startIndex;
		int indexDst = transformed.startIndex;
		int end = indexSrc + input.width;

		double total = 0;
		for( ; indexSrc < end; indexSrc++ ) {
			transformed.data[indexDst++] = total += input.data[indexSrc];
		}

		for( int y = 1; y < input.height; y++ ) {
			indexSrc = input.startIndex + input.stride*y;
			indexDst = transformed.startIndex + transformed.stride*y;
			int indexPrev = indexDst - transformed.stride;

			end = indexSrc + input.width;

			total = 0;
			for( ; indexSrc < end; indexSrc++ ) {
				total +=  input.data[indexSrc];
				transformed.data[indexDst++] = transformed.data[indexPrev++] + total;
			}
		}
	}

	public static void transform( final GrayU8 input , final GrayS32 transformed )
	{
		int indexSrc = input.startIndex;
		int indexDst = transformed.startIndex;
		int end = indexSrc + input.width;

		int total = 0;
		for( ; indexSrc < end; indexSrc++ ) {
			transformed.data[indexDst++] = total += input.data[indexSrc]& 0xFF;
		}

		for( int y = 1; y < input.height; y++ ) {
			indexSrc = input.startIndex + input.stride*y;
			indexDst = transformed.startIndex + transformed.stride*y;
			int indexPrev = indexDst - transformed.stride;

			end = indexSrc + input.width;

			total = 0;
			for( ; indexSrc < end; indexSrc++ ) {
				total +=  input.data[indexSrc]& 0xFF;
				transformed.data[indexDst++] = transformed.data[indexPrev++] + total;
			}
		}
	}

	public static void transform( final GrayS32 input , final GrayS32 transformed )
	{
		int indexSrc = input.startIndex;
		int indexDst = transformed.startIndex;
		int end = indexSrc + input.width;

		int total = 0;
		for( ; indexSrc < end; indexSrc++ ) {
			transformed.data[indexDst++] = total += input.data[indexSrc];
		}

		for( int y = 1; y < input.height; y++ ) {
			indexSrc = input.startIndex + input.stride*y;
			indexDst = transformed.startIndex + transformed.stride*y;
			int indexPrev = indexDst - transformed.stride;

			end = indexSrc + input.width;

			total = 0;
			for( ; indexSrc < end; indexSrc++ ) {
				total +=  input.data[indexSrc];
				transformed.data[indexDst++] = transformed.data[indexPrev++] + total;
			}
		}
	}

	public static void transform( final GrayS64 input , final GrayS64 transformed )
	{
		int indexSrc = input.startIndex;
		int indexDst = transformed.startIndex;
		int end = indexSrc + input.width;

		long total = 0;
		for( ; indexSrc < end; indexSrc++ ) {
			transformed.data[indexDst++] = total += input.data[indexSrc];
		}

		for( int y = 1; y < input.height; y++ ) {
			indexSrc = input.startIndex + input.stride*y;
			indexDst = transformed.startIndex + transformed.stride*y;
			int indexPrev = indexDst - transformed.stride;

			end = indexSrc + input.width;

			total = 0;
			for( ; indexSrc < end; indexSrc++ ) {
				total +=  input.data[indexSrc];
				transformed.data[indexDst++] = transformed.data[indexPrev++] + total;
			}
		}
	}

	public static float convolveSparse( GrayF32 integral , IntegralKernel kernel , int x , int y )
	{
		float ret = 0;
		int N = kernel.getNumBlocks();

		for( int i = 0; i < N; i++ ) {
			ImageRectangle r = kernel.blocks[i];
			ret += block_zero(integral,x+r.x0,y+r.y0,x+r.x1,y+r.y1)*kernel.scales[i];
		}

		return ret;
	}

	public static float block_unsafe( GrayF32 integral , int x0 , int y0 , int x1 , int y1 )
	{
		float br = integral.data[ integral.startIndex + y1*integral.stride + x1 ];
		float tr = integral.data[ integral.startIndex + y0*integral.stride + x1 ];
		float bl = integral.data[ integral.startIndex + y1*integral.stride + x0 ];
		float tl = integral.data[ integral.startIndex + y0*integral.stride + x0 ];

		return br-tr-bl+tl;
	}

	public static float block_zero( GrayF32 integral , int x0 , int y0 , int x1 , int y1 )
	{
		x0 = Math.min(x0,integral.width-1);
		y0 = Math.min(y0,integral.height-1);
		x1 = Math.min(x1,integral.width-1);
		y1 = Math.min(y1,integral.height-1);

		float br=0,tr=0,bl=0,tl=0;

		if( x1 >= 0 && y1 >= 0)
			br = integral.data[ integral.startIndex + y1*integral.stride + x1 ];
		if( y0 >= 0 && x1 >= 0)
			tr = integral.data[ integral.startIndex + y0*integral.stride + x1 ];
		if( x0 >= 0 && y1 >= 0)
			bl = integral.data[ integral.startIndex + y1*integral.stride + x0 ];
		if( x0 >= 0 && y0 >= 0)
			tl = integral.data[ integral.startIndex + y0*integral.stride + x0 ];

		return br-tr-bl+tl;
	}

	public static int convolveSparse( GrayS32 integral , IntegralKernel kernel , int x , int y )
	{
		int ret = 0;
		int N = kernel.getNumBlocks();

		for( int i = 0; i < N; i++ ) {
			ImageRectangle r = kernel.blocks[i];
			ret += block_zero(integral,x+r.x0,y+r.y0,x+r.x1,y+r.y1)*kernel.scales[i];
		}

		return ret;
	}

	public static int block_unsafe( GrayS32 integral , int x0 , int y0 , int x1 , int y1 )
	{
		int br = integral.data[ integral.startIndex + y1*integral.stride + x1 ];
		int tr = integral.data[ integral.startIndex + y0*integral.stride + x1 ];
		int bl = integral.data[ integral.startIndex + y1*integral.stride + x0 ];
		int tl = integral.data[ integral.startIndex + y0*integral.stride + x0 ];

		return br-tr-bl+tl;
	}

	public static int block_zero( GrayS32 integral , int x0 , int y0 , int x1 , int y1 )
	{
		x0 = Math.min(x0,integral.width-1);
		y0 = Math.min(y0,integral.height-1);
		x1 = Math.min(x1,integral.width-1);
		y1 = Math.min(y1,integral.height-1);

		int br=0,tr=0,bl=0,tl=0;

		if( x1 >= 0 && y1 >= 0)
			br = integral.data[ integral.startIndex + y1*integral.stride + x1 ];
		if( y0 >= 0 && x1 >= 0)
			tr = integral.data[ integral.startIndex + y0*integral.stride + x1 ];
		if( x0 >= 0 && y1 >= 0)
			bl = integral.data[ integral.startIndex + y1*integral.stride + x0 ];
		if( x0 >= 0 && y0 >= 0)
			tl = integral.data[ integral.startIndex + y0*integral.stride + x0 ];

		return br-tr-bl+tl;
	}

	public static double convolveSparse( GrayF64 integral , IntegralKernel kernel , int x , int y )
	{
		double ret = 0;
		int N = kernel.getNumBlocks();

		for( int i = 0; i < N; i++ ) {
			ImageRectangle r = kernel.blocks[i];
			ret += block_zero(integral,x+r.x0,y+r.y0,x+r.x1,y+r.y1)*kernel.scales[i];
		}

		return ret;
	}

	public static double block_unsafe( GrayF64 integral , int x0 , int y0 , int x1 , int y1 )
	{
		double br = integral.data[ integral.startIndex + y1*integral.stride + x1 ];
		double tr = integral.data[ integral.startIndex + y0*integral.stride + x1 ];
		double bl = integral.data[ integral.startIndex + y1*integral.stride + x0 ];
		double tl = integral.data[ integral.startIndex + y0*integral.stride + x0 ];

		return br-tr-bl+tl;
	}

	public static double block_zero( GrayF64 integral , int x0 , int y0 , int x1 , int y1 )
	{
		x0 = Math.min(x0,integral.width-1);
		y0 = Math.min(y0,integral.height-1);
		x1 = Math.min(x1,integral.width-1);
		y1 = Math.min(y1,integral.height-1);

		double br=0,tr=0,bl=0,tl=0;

		if( x1 >= 0 && y1 >= 0)
			br = integral.data[ integral.startIndex + y1*integral.stride + x1 ];
		if( y0 >= 0 && x1 >= 0)
			tr = integral.data[ integral.startIndex + y0*integral.stride + x1 ];
		if( x0 >= 0 && y1 >= 0)
			bl = integral.data[ integral.startIndex + y1*integral.stride + x0 ];
		if( x0 >= 0 && y0 >= 0)
			tl = integral.data[ integral.startIndex + y0*integral.stride + x0 ];

		return br-tr-bl+tl;
	}

	public static long convolveSparse( GrayS64 integral , IntegralKernel kernel , int x , int y )
	{
		long ret = 0;
		int N = kernel.getNumBlocks();

		for( int i = 0; i < N; i++ ) {
			ImageRectangle r = kernel.blocks[i];
			ret += block_zero(integral,x+r.x0,y+r.y0,x+r.x1,y+r.y1)*kernel.scales[i];
		}

		return ret;
	}

	public static long block_unsafe( GrayS64 integral , int x0 , int y0 , int x1 , int y1 )
	{
		long br = integral.data[ integral.startIndex + y1*integral.stride + x1 ];
		long tr = integral.data[ integral.startIndex + y0*integral.stride + x1 ];
		long bl = integral.data[ integral.startIndex + y1*integral.stride + x0 ];
		long tl = integral.data[ integral.startIndex + y0*integral.stride + x0 ];

		return br-tr-bl+tl;
	}

	public static long block_zero( GrayS64 integral , int x0 , int y0 , int x1 , int y1 )
	{
		x0 = Math.min(x0,integral.width-1);
		y0 = Math.min(y0,integral.height-1);
		x1 = Math.min(x1,integral.width-1);
		y1 = Math.min(y1,integral.height-1);

		long br=0,tr=0,bl=0,tl=0;

		if( x1 >= 0 && y1 >= 0)
			br = integral.data[ integral.startIndex + y1*integral.stride + x1 ];
		if( y0 >= 0 && x1 >= 0)
			tr = integral.data[ integral.startIndex + y0*integral.stride + x1 ];
		if( x0 >= 0 && y1 >= 0)
			bl = integral.data[ integral.startIndex + y1*integral.stride + x0 ];
		if( x0 >= 0 && y0 >= 0)
			tl = integral.data[ integral.startIndex + y0*integral.stride + x0 ];

		return br-tr-bl+tl;
	}


}
