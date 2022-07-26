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

package boofcv.alg.transform.wavelet.impl;

import boofcv.alg.transform.wavelet.UtilWavelet;
import boofcv.struct.border.BorderIndex1D;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.GrayS32;
import boofcv.struct.wavelet.WlBorderCoef;
import boofcv.struct.wavelet.WlCoef;
import boofcv.struct.wavelet.WlCoef_F32;
import boofcv.struct.wavelet.WlCoef_I32;

import javax.annotation.Generated;

/**
 * <p>
 * Performs the wavelet transform just around the image border. Should be called in conjunction
 * with {@link ImplWaveletTransformInner} or similar functions. Must be called after the inner
 * portion has been computed because the "inner" functions modify the border during the inverse
 * transform.
 * </p>
 *
 * <p>
 * For the inverse transform the inner transform must be called before the border is computed.
 * Due to how the inverse is computed some of the output values will be added to border. The values
 * computed in these inverse functions add to that.
 * </p>
 * 
 * <p>
 * DO NOT MODIFY: This class was automatically generated by {@link boofcv.alg.transform.wavelet.impl.GenerateImplWaveletTransformBorder}
 * </p>
 *
 * @author Peter Abeles
 */
@Generated("boofcv.alg.transform.wavelet.impl.GenerateImplWaveletTransformBorder")
@SuppressWarnings({"ForLoopReplaceableByForEach","NarrowingCompoundAssignment"})
public class ImplWaveletTransformBorder {

	public static void horizontal(BorderIndex1D border , WlCoef_F32 coefficients , GrayF32 input , GrayF32 output )
	{
		final int offsetA = coefficients.offsetScaling;
		final int offsetB = coefficients.offsetWavelet;
		final float[] alpha = coefficients.scaling;
		final float[] beta = coefficients.wavelet;

		border.setLength(input.width + input.width%2);

		final boolean isLarger = output.width > input.width;
		final int width = input.width+input.width%2;
		final int height = input.height;
		final int lowerBorder = UtilWavelet.borderForwardLower(coefficients);
		final int upperBorder = input.width - UtilWavelet.borderForwardUpper(coefficients,input.width);

		for( int y = 0; y < height; y++ ) {
			for( int x = 0; x < lowerBorder; x += 2 ) {
				float scale = 0;
				float wavelet = 0;

				for( int i = 0; i < alpha.length; i++ ) {
					int xx = border.getIndex(x+i+offsetA);
					if( isLarger && xx >= input.width )
						continue;
					scale += input.get(xx,y)*alpha[i];
				}
				for( int i = 0; i < beta.length; i++ ) {
					int xx = border.getIndex(x+i+offsetB);
					if( isLarger && xx >= input.width )
						continue;
					wavelet += input.get(xx,y)*beta[i];
				}

				int outX = x/2;

				output.set(outX,y,scale);
				output.set(output.width/2 + outX , y , wavelet );
			}
			for( int x = upperBorder; x < width; x += 2 ) {
				float scale = 0;
				float wavelet = 0;

				for( int i = 0; i < alpha.length; i++ ) {
					int xx = border.getIndex(x+i+offsetA);
					if( isLarger && xx >= input.width )
						continue;
					scale += input.get(xx,y)*alpha[i];
				}
				for( int i = 0; i < beta.length; i++ ) {
					int xx = border.getIndex(x+i+offsetB);
					if( isLarger && xx >= input.width )
						continue;
					wavelet += input.get(xx,y)*beta[i];
				}

				int outX = x/2;

				output.set(outX,y,scale);
				output.set(output.width/2 + outX , y , wavelet );
			}
		}
	}

	public static void vertical(BorderIndex1D border , WlCoef_F32 coefficients , GrayF32 input , GrayF32 output )
	{
		final int offsetA = coefficients.offsetScaling;
		final int offsetB = coefficients.offsetWavelet;
		final float[] alpha = coefficients.scaling;
		final float[] beta = coefficients.wavelet;

		border.setLength(input.height + input.height%2);

		final boolean isLarger = output.height > input.height;
		final int width = input.width;
		final int height = input.height+input.height%2;
		final int lowerBorder = UtilWavelet.borderForwardLower(coefficients);
		final int upperBorder = input.height - UtilWavelet.borderForwardUpper(coefficients,input.height);

		for( int x = 0; x < width; x++) {
			for( int y = 0; y < lowerBorder; y += 2 ) {
				float scale = 0;
				float wavelet = 0;

				for( int i = 0; i < alpha.length; i++ ) {
					int yy = border.getIndex(y+i+offsetA);
					if( isLarger && yy >= input.height )
						continue;
					scale += input.get(x,yy)*alpha[i];
				}
				for( int i = 0; i < beta.length; i++ ) {
					int yy = border.getIndex(y+i+offsetB);
					if( isLarger && yy >= input.height )
						continue;
					wavelet += input.get(x,yy)*beta[i];
				}

				int outY = y/2;

				output.set(x , outY,scale);
				output.set(x , output.height/2 + outY , wavelet );
			}

			for( int y = upperBorder; y < height; y += 2 ) {
				float scale = 0;
				float wavelet = 0;

				for( int i = 0; i < alpha.length; i++ ) {
					int yy = border.getIndex(y+i+offsetA);
					if( isLarger && yy >= input.height )
						continue;
					scale += input.get(x,yy)*alpha[i];
				}
				for( int i = 0; i < beta.length; i++ ) {
					int yy = border.getIndex(y+i+offsetB);
					if( isLarger && yy >= input.height )
						continue;
					wavelet += input.get(x,yy)*beta[i];
				}

				int outY = y/2;

				output.set(x , outY,scale);
				output.set(x , output.height/2 + outY , wavelet );
			}
		}
	}

	public static void horizontalInverse(BorderIndex1D border , WlBorderCoef<WlCoef_F32> desc , GrayF32 input , GrayF32 output )
	{
		float []trends = new float[ input.width ];
		float []details = new float[ input.width ];

		final int height = output.height;
		final int paddedWidth = output.width + output.width%2;

		WlCoef inner = desc.getInnerCoefficients();
		// need to convolve coefficients that influence the ones being updated
		int lowerExtra = -Math.min(inner.offsetScaling,inner.offsetWavelet);
		int upperExtra = Math.max(inner.getScalingLength()+inner.offsetScaling,inner.getWaveletLength()+inner.offsetWavelet);
		lowerExtra += lowerExtra%2;
		upperExtra += upperExtra%2;

		int lowerBorder = (UtilWavelet.borderInverseLower(desc,border)+lowerExtra)/2;
		int upperBorder = (UtilWavelet.borderInverseUpper(desc,border,output.width)+upperExtra)/2;

		boolean isLarger = input.width >= output.width;
		
		// where updated wavelet values are stored
		int lowerCompute = lowerBorder*2-lowerExtra;
		int upperCompute = upperBorder*2-upperExtra;

		int indexes[] = new int[lowerBorder+upperBorder];
		for( int i = 0; i < lowerBorder; i++ )
			indexes[i] = i*2;
		for( int i = lowerBorder; i < indexes.length; i++ )
			indexes[i] = paddedWidth-(indexes.length-i)*2;

		border.setLength(output.width+output.width%2);

		WlCoef_F32 coefficients;

		for( int y = 0; y < height; y++ ) {

			// initialize details and trends arrays
			for( int i = 0; i < indexes.length; i++ ) {
				int x = indexes[i];
				details[x] = 0; trends[x] = 0;
				x++;
				details[x] = 0; trends[x] = 0;
			}

			for( int i = 0; i < indexes.length; i++ ) {
				int x = indexes[i];
				float a = input.get(x/2,y);
				float d = input.get(input.width/2+x/2,y);

				if( x < lowerBorder ) {
					coefficients = desc.getBorderCoefficients(x);
				} else if( x >= upperBorder ) {
					coefficients = desc.getBorderCoefficients(x-paddedWidth);
				} else {
					coefficients = desc.getInnerCoefficients();
				}

				final int offsetA = coefficients.offsetScaling;
				final int offsetB = coefficients.offsetWavelet;
				final float[] alpha = coefficients.scaling;
				final float[] beta = coefficients.wavelet;

				// add the trend
				for( int j = 0; j < alpha.length; j++ ) {
					// if an odd image don't update the outer edge
					int xx = border.getIndex(x+offsetA+j);
					if( isLarger && xx >= output.width )
						continue;
					trends[xx] += a*alpha[j];
				}

				// add the detail signal
				for( int j = 0; j < beta.length; j++ ) {
					int xx = border.getIndex(x+offsetB+j);
					if( isLarger && xx >= output.width )
						continue;
					details[xx] += d*beta[j];
				}
			}

			int indexDst = output.startIndex + y*output.stride;
			for( int x = 0; x < lowerCompute; x++ ) {
				output.data[ indexDst + x ] = (trends[x] + details[x]);
			}
			for( int x = paddedWidth-upperCompute; x < output.width; x++) {
				output.data[ indexDst + x ] = (trends[x] + details[x]);
			}
		}
	}

	public static void verticalInverse(BorderIndex1D border , WlBorderCoef<WlCoef_F32> desc , GrayF32 input , GrayF32 output )
	{
		float []trends = new float[ input.height ];
		float []details = new float[ input.height ];

		final int width = output.width;
		final int paddedHeight = output.height + output.height%2;

		WlCoef inner = desc.getInnerCoefficients();
		// need to convolve coefficients that influence the ones being updated
		int lowerExtra = -Math.min(inner.offsetScaling,inner.offsetWavelet);
		int upperExtra = Math.max(inner.getScalingLength()+inner.offsetScaling,inner.getWaveletLength()+inner.offsetWavelet);
		lowerExtra += lowerExtra%2;
		upperExtra += upperExtra%2;

		int lowerBorder = (UtilWavelet.borderInverseLower(desc,border)+lowerExtra)/2;
		int upperBorder = (UtilWavelet.borderInverseUpper(desc,border,output.height)+upperExtra)/2;

		boolean isLarger = input.height >= output.height;
		
		// where updated wavelet values are stored
		int lowerCompute = lowerBorder*2-lowerExtra;
		int upperCompute = upperBorder*2-upperExtra;

		int indexes[] = new int[lowerBorder+upperBorder];
		for( int i = 0; i < lowerBorder; i++ )
			indexes[i] = i*2;
		for( int i = lowerBorder; i < indexes.length; i++ )
			indexes[i] = paddedHeight-(indexes.length-i)*2;

		border.setLength(output.height+output.height%2);

		WlCoef_F32 coefficients;

		for( int x = 0; x < width; x++ ) {

			// initialize details and trends arrays
			for( int i = 0; i < indexes.length; i++ ) {
				int y = indexes[i];
				details[y] = 0; trends[y] = 0;
				y++;
				details[y] = 0; trends[y] = 0;
			}

			for( int i = 0; i < indexes.length; i++ ) {
				int y = indexes[i];
				float a = input.get(x,y/2);
				float d = input.get(x,input.height/2+y/2);

				if( y < lowerBorder ) {
					coefficients = desc.getBorderCoefficients(y);
				} else if( y >= upperBorder ) {
					coefficients = desc.getBorderCoefficients(y-paddedHeight);
				} else {
					coefficients = desc.getInnerCoefficients();
				}

				final int offsetA = coefficients.offsetScaling;
				final int offsetB = coefficients.offsetWavelet;
				final float[] alpha = coefficients.scaling;
				final float[] beta = coefficients.wavelet;

				// add the trend
				for( int j = 0; j < alpha.length; j++ ) {
					// if an odd image don't update the outer edge
					int yy = border.getIndex(y+offsetA+j);
					if( isLarger && yy >= output.height )
						continue;
					trends[yy] += a*alpha[j];
				}

				// add the detail signal
				for( int j = 0; j < beta.length; j++ ) {
					int yy = border.getIndex(y+offsetB+j);
					if( isLarger && yy >= output.height )
						continue;
					details[yy] += d*beta[j];
				}
			}

			int indexDst = output.startIndex + x;
			for( int y = 0; y < lowerCompute; y++ ) {
				output.data[ indexDst + y*output.stride ] = (trends[y] + details[y]);
			}
			for( int y = paddedHeight-upperCompute; y < output.height; y++) {
				output.data[ indexDst + y*output.stride ] = (trends[y] + details[y]);
			}
		}
	}

	public static void horizontal(BorderIndex1D border , WlCoef_I32 coefficients , GrayS32 input , GrayS32 output )
	{
		final int offsetA = coefficients.offsetScaling;
		final int offsetB = coefficients.offsetWavelet;
		final int[] alpha = coefficients.scaling;
		final int[] beta = coefficients.wavelet;

		border.setLength(input.width + input.width%2);

		final boolean isLarger = output.width > input.width;
		final int width = input.width+input.width%2;
		final int height = input.height;
		final int lowerBorder = UtilWavelet.borderForwardLower(coefficients);
		final int upperBorder = input.width - UtilWavelet.borderForwardUpper(coefficients,input.width);

		for( int y = 0; y < height; y++ ) {
			for( int x = 0; x < lowerBorder; x += 2 ) {
				int scale = 0;
				int wavelet = 0;

				for( int i = 0; i < alpha.length; i++ ) {
					int xx = border.getIndex(x+i+offsetA);
					if( isLarger && xx >= input.width )
						continue;
					scale += input.get(xx,y)*alpha[i];
				}
				for( int i = 0; i < beta.length; i++ ) {
					int xx = border.getIndex(x+i+offsetB);
					if( isLarger && xx >= input.width )
						continue;
					wavelet += input.get(xx,y)*beta[i];
				}

				scale = 2*scale/coefficients.denominatorScaling;
				wavelet = 2*wavelet/coefficients.denominatorWavelet;

				int outX = x/2;

				output.set(outX,y,scale);
				output.set(output.width/2 + outX , y , wavelet );
			}
			for( int x = upperBorder; x < width; x += 2 ) {
				int scale = 0;
				int wavelet = 0;

				for( int i = 0; i < alpha.length; i++ ) {
					int xx = border.getIndex(x+i+offsetA);
					if( isLarger && xx >= input.width )
						continue;
					scale += input.get(xx,y)*alpha[i];
				}
				for( int i = 0; i < beta.length; i++ ) {
					int xx = border.getIndex(x+i+offsetB);
					if( isLarger && xx >= input.width )
						continue;
					wavelet += input.get(xx,y)*beta[i];
				}

				int outX = x/2;

				scale = 2*scale/coefficients.denominatorScaling;
				wavelet = 2*wavelet/coefficients.denominatorWavelet;

				output.set(outX,y,scale);
				output.set(output.width/2 + outX , y , wavelet );
			}
		}
	}

	public static void vertical(BorderIndex1D border , WlCoef_I32 coefficients , GrayS32 input , GrayS32 output )
	{
		final int offsetA = coefficients.offsetScaling;
		final int offsetB = coefficients.offsetWavelet;
		final int[] alpha = coefficients.scaling;
		final int[] beta = coefficients.wavelet;

		border.setLength(input.height + input.height%2);

		final boolean isLarger = output.height > input.height;
		final int width = input.width;
		final int height = input.height+input.height%2;
		final int lowerBorder = UtilWavelet.borderForwardLower(coefficients);
		final int upperBorder = input.height - UtilWavelet.borderForwardUpper(coefficients,input.height);

		for( int x = 0; x < width; x++) {
			for( int y = 0; y < lowerBorder; y += 2 ) {
				int scale = 0;
				int wavelet = 0;

				for( int i = 0; i < alpha.length; i++ ) {
					int yy = border.getIndex(y+i+offsetA);
					if( isLarger && yy >= input.height )
						continue;
					scale += input.get(x,yy)*alpha[i];
				}
				for( int i = 0; i < beta.length; i++ ) {
					int yy = border.getIndex(y+i+offsetB);
					if( isLarger && yy >= input.height )
						continue;
					wavelet += input.get(x,yy)*beta[i];
				}

				int outY = y/2;

				scale = 2*scale/coefficients.denominatorScaling;
				wavelet = 2*wavelet/coefficients.denominatorWavelet;

				output.set(x , outY,scale);
				output.set(x , output.height/2 + outY , wavelet );
			}

			for( int y = upperBorder; y < height; y += 2 ) {
				int scale = 0;
				int wavelet = 0;

				for( int i = 0; i < alpha.length; i++ ) {
					int yy = border.getIndex(y+i+offsetA);
					if( isLarger && yy >= input.height )
						continue;
					scale += input.get(x,yy)*alpha[i];
				}
				for( int i = 0; i < beta.length; i++ ) {
					int yy = border.getIndex(y+i+offsetB);
					if( isLarger && yy >= input.height )
						continue;
					wavelet += input.get(x,yy)*beta[i];
				}

				int outY = y/2;

				scale = 2*scale/coefficients.denominatorScaling;
				wavelet = 2*wavelet/coefficients.denominatorWavelet;

				output.set(x , outY,scale);
				output.set(x , output.height/2 + outY , wavelet );
			}
		}
	}

	public static void horizontalInverse(BorderIndex1D border , WlBorderCoef<WlCoef_I32> desc , GrayS32 input , GrayS32 output )
	{
		int []trends = new int[ input.width ];
		int []details = new int[ input.width ];

		final int height = output.height;
		final int paddedWidth = output.width + output.width%2;

		WlCoef inner = desc.getInnerCoefficients();
		// need to convolve coefficients that influence the ones being updated
		int lowerExtra = -Math.min(inner.offsetScaling,inner.offsetWavelet);
		int upperExtra = Math.max(inner.getScalingLength()+inner.offsetScaling,inner.getWaveletLength()+inner.offsetWavelet);
		lowerExtra += lowerExtra%2;
		upperExtra += upperExtra%2;

		int lowerBorder = (UtilWavelet.borderInverseLower(desc,border)+lowerExtra)/2;
		int upperBorder = (UtilWavelet.borderInverseUpper(desc,border,output.width)+upperExtra)/2;

		boolean isLarger = input.width >= output.width;
		
		// where updated wavelet values are stored
		int lowerCompute = lowerBorder*2-lowerExtra;
		int upperCompute = upperBorder*2-upperExtra;

		int indexes[] = new int[lowerBorder+upperBorder];
		for( int i = 0; i < lowerBorder; i++ )
			indexes[i] = i*2;
		for( int i = lowerBorder; i < indexes.length; i++ )
			indexes[i] = paddedWidth-(indexes.length-i)*2;

		border.setLength(output.width+output.width%2);

		WlCoef_I32 coefficients = desc.getInnerCoefficients();
		final int e = coefficients.denominatorScaling*2;
		final int f = coefficients.denominatorWavelet*2;
		final int ef = e*f;
		final int ef2 = ef/2;

		for( int y = 0; y < height; y++ ) {

			// initialize details and trends arrays
			for( int i = 0; i < indexes.length; i++ ) {
				int x = indexes[i];
				details[x] = 0; trends[x] = 0;
				x++;
				details[x] = 0; trends[x] = 0;
			}

			for( int i = 0; i < indexes.length; i++ ) {
				int x = indexes[i];
				float a = input.get(x/2,y);
				float d = input.get(input.width/2+x/2,y);

				if( x < lowerBorder ) {
					coefficients = desc.getBorderCoefficients(x);
				} else if( x >= upperBorder ) {
					coefficients = desc.getBorderCoefficients(x-paddedWidth);
				} else {
					coefficients = desc.getInnerCoefficients();
				}

				final int offsetA = coefficients.offsetScaling;
				final int offsetB = coefficients.offsetWavelet;
				final int[] alpha = coefficients.scaling;
				final int[] beta = coefficients.wavelet;

				// add the trend
				for( int j = 0; j < alpha.length; j++ ) {
					// if an odd image don't update the outer edge
					int xx = border.getIndex(x+offsetA+j);
					if( isLarger && xx >= output.width )
						continue;
					trends[xx] += a*alpha[j];
				}

				// add the detail signal
				for( int j = 0; j < beta.length; j++ ) {
					int xx = border.getIndex(x+offsetB+j);
					if( isLarger && xx >= output.width )
						continue;
					details[xx] += d*beta[j];
				}
			}

			int indexDst = output.startIndex + y*output.stride;
			for( int x = 0; x < lowerCompute; x++ ) {
				output.data[ indexDst + x ] = UtilWavelet.round(trends[x]*f + details[x]*e , ef2 , ef);
			}
			for( int x = paddedWidth-upperCompute; x < output.width; x++) {
				output.data[ indexDst + x ] = UtilWavelet.round(trends[x]*f + details[x]*e , ef2 , ef);
			}
		}
	}

	public static void verticalInverse(BorderIndex1D border , WlBorderCoef<WlCoef_I32> desc , GrayS32 input , GrayS32 output )
	{
		int []trends = new int[ input.height ];
		int []details = new int[ input.height ];

		final int width = output.width;
		final int paddedHeight = output.height + output.height%2;

		WlCoef inner = desc.getInnerCoefficients();
		// need to convolve coefficients that influence the ones being updated
		int lowerExtra = -Math.min(inner.offsetScaling,inner.offsetWavelet);
		int upperExtra = Math.max(inner.getScalingLength()+inner.offsetScaling,inner.getWaveletLength()+inner.offsetWavelet);
		lowerExtra += lowerExtra%2;
		upperExtra += upperExtra%2;

		int lowerBorder = (UtilWavelet.borderInverseLower(desc,border)+lowerExtra)/2;
		int upperBorder = (UtilWavelet.borderInverseUpper(desc,border,output.height)+upperExtra)/2;

		boolean isLarger = input.height >= output.height;
		
		// where updated wavelet values are stored
		int lowerCompute = lowerBorder*2-lowerExtra;
		int upperCompute = upperBorder*2-upperExtra;

		int indexes[] = new int[lowerBorder+upperBorder];
		for( int i = 0; i < lowerBorder; i++ )
			indexes[i] = i*2;
		for( int i = lowerBorder; i < indexes.length; i++ )
			indexes[i] = paddedHeight-(indexes.length-i)*2;

		border.setLength(output.height+output.height%2);

		WlCoef_I32 coefficients = desc.getInnerCoefficients();
		final int e = coefficients.denominatorScaling*2;
		final int f = coefficients.denominatorWavelet*2;
		final int ef = e*f;
		final int ef2 = ef/2;

		for( int x = 0; x < width; x++ ) {

			// initialize details and trends arrays
			for( int i = 0; i < indexes.length; i++ ) {
				int y = indexes[i];
				details[y] = 0; trends[y] = 0;
				y++;
				details[y] = 0; trends[y] = 0;
			}

			for( int i = 0; i < indexes.length; i++ ) {
				int y = indexes[i];
				float a = input.get(x,y/2);
				float d = input.get(x,input.height/2+y/2);

				if( y < lowerBorder ) {
					coefficients = desc.getBorderCoefficients(y);
				} else if( y >= upperBorder ) {
					coefficients = desc.getBorderCoefficients(y-paddedHeight);
				} else {
					coefficients = desc.getInnerCoefficients();
				}

				final int offsetA = coefficients.offsetScaling;
				final int offsetB = coefficients.offsetWavelet;
				final int[] alpha = coefficients.scaling;
				final int[] beta = coefficients.wavelet;

				// add the trend
				for( int j = 0; j < alpha.length; j++ ) {
					// if an odd image don't update the outer edge
					int yy = border.getIndex(y+offsetA+j);
					if( isLarger && yy >= output.height )
						continue;
					trends[yy] += a*alpha[j];
				}

				// add the detail signal
				for( int j = 0; j < beta.length; j++ ) {
					int yy = border.getIndex(y+offsetB+j);
					if( isLarger && yy >= output.height )
						continue;
					details[yy] += d*beta[j];
				}
			}

			int indexDst = output.startIndex + x;
			for( int y = 0; y < lowerCompute; y++ ) {
				output.data[ indexDst + y*output.stride ] = UtilWavelet.round(trends[y]*f + details[y]*e , ef2 , ef);
			}
			for( int y = paddedHeight-upperCompute; y < output.height; y++) {
				output.data[ indexDst + y*output.stride ] = UtilWavelet.round(trends[y]*f + details[y]*e , ef2 , ef);
			}
		}
	}


}
