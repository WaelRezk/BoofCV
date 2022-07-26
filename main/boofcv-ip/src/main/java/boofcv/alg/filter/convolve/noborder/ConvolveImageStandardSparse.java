/*
 * Copyright (c) 2021, Peter Abeles. All Rights Reserved.
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

package boofcv.alg.filter.convolve.noborder;

import boofcv.struct.convolve.Kernel1D_F32;
import boofcv.struct.convolve.Kernel1D_S32;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.GrayS16;
import boofcv.struct.image.GrayU8;

import javax.annotation.Generated;

/**
 *
 * <p>
 * General implementation of {@link boofcv.alg.filter.convolve.ConvolveImageNoBorderSparse}.
 * </p>
 *
 * <p>
 * DO NOT MODIFY. Auto generated by {@link GenerateConvolveStandardSparse}.
 * </p>
 *
 * @author Peter Abeles
 */
@Generated({"boofcv.alg.filter.convolve.noborder.GenerateConvolveStandardSparse"})
public class ConvolveImageStandardSparse {

	public static float convolve(Kernel1D_F32 horizontal, Kernel1D_F32 vertical,
								 GrayF32 input, int c_x , int c_y, float storage[] )
	{
		int widthH = horizontal.getWidth();
		int widthV = vertical.getWidth();
		int offsetH = horizontal.getOffset();
		int offsetV = vertical.getOffset();

		// convolve horizontally first

		for( int i = 0; i < widthV; i++ ) {
			int indexImg = input.startIndex + (i+c_y-offsetV)*input.stride + c_x-offsetH;

			float total = 0;
			for( int j = 0; j < widthH; j++ ,indexImg++) {
				total += (input.data[indexImg])*horizontal.data[j];
			}
			storage[i] = total;
		}

		// convolve vertically
		float total = 0;
		for( int i = 0; i < widthV; i++ ) {
			total += storage[i]*vertical.data[i];
		}
		return total;
	}

	public static int convolve(Kernel1D_S32 horizontal, Kernel1D_S32 vertical,
							   GrayU8 input, int c_x , int c_y, int storage[] )
	{
		int widthH = horizontal.getWidth();
		int widthV = vertical.getWidth();
		int offsetH = horizontal.getOffset();
		int offsetV = vertical.getOffset();

		// convolve horizontally first

		for( int i = 0; i < widthV; i++ ) {
			int indexImg = input.startIndex + (i+c_y-offsetV)*input.stride + c_x-offsetH;

			int total = 0;
			for( int j = 0; j < widthH; j++ ,indexImg++) {
				total += (input.data[indexImg] & 0xFF)*horizontal.data[j];
			}
			storage[i] = total;
		}

		// convolve vertically
		int total = 0;
		for( int i = 0; i < widthV; i++ ) {
			total += storage[i]*vertical.data[i];
		}
		return total;
	}

	public static int convolve(Kernel1D_S32 horizontal, Kernel1D_S32 vertical,
							   GrayU8 input, int c_x , int c_y, int storage[] ,
							   int divisorHorizontal ,
							   int divisorVertical )
	{
		int widthH = horizontal.getWidth();
		int widthV = vertical.getWidth();
		int offsetH = horizontal.getOffset();
		int offsetV = vertical.getOffset();
		int halfHorizontal = divisorHorizontal/2;

		// convolve horizontally first

		for( int i = 0; i < widthV; i++ ) {
			int indexImg = input.startIndex + (i+c_y-offsetV)*input.stride + c_x-offsetH;

			int total = 0;
			for( int j = 0; j < widthH; j++ ,indexImg++) {
				total += (input.data[indexImg] & 0xFF)*horizontal.data[j];
			}
			storage[i] = (total + halfHorizontal)/divisorHorizontal;
		}

		// convolve vertically
		int total = 0;
		for( int i = 0; i < widthV; i++ ) {
			total += storage[i]*vertical.data[i];
		}
		return (total + divisorVertical/2)/divisorVertical;
	}

	public static int convolve(Kernel1D_S32 horizontal, Kernel1D_S32 vertical,
							   GrayS16 input, int c_x , int c_y, int storage[] )
	{
		int widthH = horizontal.getWidth();
		int widthV = vertical.getWidth();
		int offsetH = horizontal.getOffset();
		int offsetV = vertical.getOffset();

		// convolve horizontally first

		for( int i = 0; i < widthV; i++ ) {
			int indexImg = input.startIndex + (i+c_y-offsetV)*input.stride + c_x-offsetH;

			int total = 0;
			for( int j = 0; j < widthH; j++ ,indexImg++) {
				total += (input.data[indexImg])*horizontal.data[j];
			}
			storage[i] = total;
		}

		// convolve vertically
		int total = 0;
		for( int i = 0; i < widthV; i++ ) {
			total += storage[i]*vertical.data[i];
		}
		return total;
	}

	public static int convolve(Kernel1D_S32 horizontal, Kernel1D_S32 vertical,
							   GrayS16 input, int c_x , int c_y, int storage[] ,
							   int divisorHorizontal ,
							   int divisorVertical )
	{
		int widthH = horizontal.getWidth();
		int widthV = vertical.getWidth();
		int offsetH = horizontal.getOffset();
		int offsetV = vertical.getOffset();
		int halfHorizontal = divisorHorizontal/2;

		// convolve horizontally first

		for( int i = 0; i < widthV; i++ ) {
			int indexImg = input.startIndex + (i+c_y-offsetV)*input.stride + c_x-offsetH;

			int total = 0;
			for( int j = 0; j < widthH; j++ ,indexImg++) {
				total += (input.data[indexImg])*horizontal.data[j];
			}
			storage[i] = (total + halfHorizontal)/divisorHorizontal;
		}

		// convolve vertically
		int total = 0;
		for( int i = 0; i < widthV; i++ ) {
			total += storage[i]*vertical.data[i];
		}
		return (total + divisorVertical/2)/divisorVertical;
	}

}
