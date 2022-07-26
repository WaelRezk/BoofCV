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

package boofcv.core.image;

import boofcv.alg.interpolate.InterpolatePixelS;
import boofcv.struct.distort.PixelTransform;
import boofcv.struct.image.ImageGray;
import georegression.struct.point.Point2D_F32;

/**
 * Implementation of {@link GImageGray} that applies a {@link PixelTransform} then
 * {@link InterpolatePixelS interpolates} to get the pixel's value.
 *
 * @author Peter Abeles
 */
public class GImageGrayDistorted<T extends ImageGray<T>> implements GImageGray {

	PixelTransform<Point2D_F32> transform;
	InterpolatePixelS<T> interpolate;

	Point2D_F32 distorted = new Point2D_F32();
	int inputWidth, inputHeight;

	public GImageGrayDistorted( PixelTransform<Point2D_F32> transform,
								InterpolatePixelS<T> interpolate ) {
		this.transform = transform;
		this.interpolate = interpolate;
	}

	@Override
	public void wrap( ImageGray image ) {
		interpolate.setImage((T)image);

		inputWidth = image.getWidth();
		inputHeight = image.getHeight();
	}

	@Override
	public int getWidth() {
		return inputWidth;
	}

	@Override
	public int getHeight() {
		return inputHeight;
	}

	@Override
	public boolean isFloatingPoint() {
		return true;
	}

	@Override
	public Number get( int x, int y ) {
		transform.compute(x, y, distorted);
		return interpolate.get(distorted.x, distorted.y);
	}

	@Override
	public void set( int x, int y, Number num ) {
		throw new IllegalArgumentException("set is not supported");
	}

	@Override
	public double unsafe_getD( int x, int y ) {
		transform.compute(x, y, distorted);
		return interpolate.get(distorted.x, distorted.y);
	}

	@Override
	public float unsafe_getF( int x, int y ) {
		transform.compute(x, y, distorted);
		return interpolate.get(distorted.x, distorted.y);
	}

	@Override
	public void set( int index, float value ) {
		throw new IllegalArgumentException("set is not supported");
	}

	@Override
	public float getF( int index ) {
		throw new IllegalArgumentException("getF is not supported");
	}

	@Override
	public ImageGray getImage() {
		throw new IllegalArgumentException("getImage() is not supported");
	}

	@Override
	public Class getImageType() {
		throw new IllegalArgumentException("getImageType() is not supported");
	}
}
