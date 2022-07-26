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

package boofcv.abst.feature.describe;

import boofcv.alg.feature.describe.DescribePointBriefSO;
import boofcv.struct.feature.TupleDesc_B;
import boofcv.struct.image.ImageGray;
import boofcv.struct.image.ImageType;

/**
 * Wrapper around {@link DescribePointBriefSO} for {@link DescribePointRadiusAngle}
 *
 * @author Peter Abeles
 */
public class DescribeBriefSO_RadiusAngle<T extends ImageGray<T>> implements DescribePointRadiusAngle<T, TupleDesc_B> {

	int length;
	DescribePointBriefSO<T> alg;
	ImageType<T> imageType;

	public DescribeBriefSO_RadiusAngle( DescribePointBriefSO<T> alg, Class<T> imageType ) {
		this.alg = alg;
		this.length = alg.getDefinition().getLength();
		this.imageType = ImageType.single(imageType);
	}

	@Override
	public TupleDesc_B createDescription() {
		return new TupleDesc_B(length);
	}

	@Override
	public void setImage( T image ) {
		alg.setImage(image);
	}

	@Override
	public boolean process( double x, double y, double orientation, double radius, TupleDesc_B storage ) {
		alg.process((float)x, (float)y, (float)orientation, (float)radius, storage);
		return true;
	}

	@Override
	public boolean isScalable() {
		return true;
	}

	@Override
	public boolean isOriented() {
		return true;
	}

	@Override
	public ImageType<T> getImageType() {
		return imageType;
	}

	@Override
	public Class<TupleDesc_B> getDescriptionType() {
		return TupleDesc_B.class;
	}

	@Override
	public double getCanonicalWidth() {
		return alg.getCanonicalWidth();
	}
}
