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

package boofcv.alg.transform.pyramid.impl;

import boofcv.alg.interpolate.InterpolatePixelS;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.GrayU8;

//CONCURRENT_INLINE import boofcv.concurrency.BoofConcurrency;

/**
 * <p>
 * Image type specific implementations of functions in {@link boofcv.alg.transform.pyramid.PyramidOps}.
 * </p>
 *
 * <p>
 * DO NOT MODIFY. This code was automatically generated by GenerateImplPyramidOps.
 * <p>
 *
 * @author Peter Abeles
 */
@SuppressWarnings("Duplicates")
public class ImplPyramidOps {

	/**
	 * Scales an image up using interpolation
	 */
	public static void scaleImageUp(GrayF32 input , GrayF32 output , int scale ,
					  InterpolatePixelS<GrayF32> interp ) {
		output.reshape(input.width*scale,input.height*scale);

		float fdiv = 1/(float)scale;
		interp.setImage(input);

		//CONCURRENT_BELOW BoofConcurrency.loopFor(0, output.height, y -> {
		for (int y = 0; y < output.height; y++) {
			float inputY = y*fdiv;
			int indexOutput = output.getIndex(0,y);

			for (int x = 0; x < output.width; x++) {
				float inputX = x*fdiv;

				output.data[indexOutput++] = interp.get(inputX,inputY);
			}
		}
		//CONCURRENT_ABOVE });
	}

	/**
	 * Scales down the input by a factor of 2. Every other pixel along both axises is skipped.
	 */
	public static void scaleDown2( GrayF32 input , GrayF32 output ) {
		
		output.reshape(input.width / 2, input.height / 2);

		//CONCURRENT_BELOW BoofConcurrency.loopFor(0, output.height, y -> {
		for (int y = 0; y < output.height; y++) {
			int indexInput = 2*y*input.stride;
			int indexOutput = y*output.stride;
			for (int x = 0; x < output.width; x++,indexInput+=2) {
				output.data[indexOutput++] = input.data[indexInput];
			}
		}
		//CONCURRENT_ABOVE });
	}

	/**
	 * Scales an image up using interpolation
	 */
	public static void scaleImageUp(GrayU8 input , GrayU8 output , int scale ,
					  InterpolatePixelS<GrayU8> interp ) {
		output.reshape(input.width*scale,input.height*scale);

		float fdiv = 1/(float)scale;
		interp.setImage(input);

		//CONCURRENT_BELOW BoofConcurrency.loopFor(0, output.height, y -> {
		for (int y = 0; y < output.height; y++) {
			float inputY = y*fdiv;
			int indexOutput = output.getIndex(0,y);

			for (int x = 0; x < output.width; x++) {
				float inputX = x*fdiv;

				output.data[indexOutput++] = (byte)interp.get(inputX,inputY);
			}
		}
		//CONCURRENT_ABOVE });
	}

	/**
	 * Scales down the input by a factor of 2. Every other pixel along both axises is skipped.
	 */
	public static void scaleDown2( GrayU8 input , GrayU8 output ) {
		
		output.reshape(input.width / 2, input.height / 2);

		//CONCURRENT_BELOW BoofConcurrency.loopFor(0, output.height, y -> {
		for (int y = 0; y < output.height; y++) {
			int indexInput = 2*y*input.stride;
			int indexOutput = y*output.stride;
			for (int x = 0; x < output.width; x++,indexInput+=2) {
				output.data[indexOutput++] = input.data[indexInput];
			}
		}
		//CONCURRENT_ABOVE });
	}


}