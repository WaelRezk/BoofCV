/*
 * Copyright (c) 2020, Peter Abeles. All Rights Reserved.
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

package boofcv.alg.feature.detect.line;

import boofcv.abst.feature.detect.extract.ConfigExtract;
import boofcv.abst.feature.detect.extract.NonMaxSuppression;
import boofcv.factory.feature.detect.extract.FactoryFeatureExtractor;
import boofcv.testing.BoofStandardJUnit;
import org.junit.jupiter.api.Nested;

/**
 * @author Peter Abeles
 */
public class TestHoughParametersPolar extends BoofStandardJUnit {

	@Nested
	class Gradient extends CommonHoughGradientChecks {
		@Override
		HoughTransformGradient createAlgorithm( Class derivType ) {
			NonMaxSuppression extractor = FactoryFeatureExtractor.nonmax(new ConfigExtract(4, 5, 0, true));
			HoughTransformParameters hough = new HoughParametersPolar(0.5,180);
			return new HoughTransformGradient(extractor,hough,derivType);
		}
	}

	@Nested
	class Gradient_MT extends CommonHoughGradientChecks {
		@Override
		HoughTransformGradient createAlgorithm( Class derivType ) {
			NonMaxSuppression extractor = FactoryFeatureExtractor.nonmax(new ConfigExtract(4, 5, 0, true));
			HoughTransformParameters hough = new HoughParametersPolar(0.5,180);
			return new HoughTransformGradient_MT(extractor,hough,derivType);
		}
	}

	@Nested
	class Binary extends CommonHoughBinaryChecks {

		@Override
		HoughTransformBinary createAlgorithm() {
			NonMaxSuppression extractor = FactoryFeatureExtractor.nonmax(new ConfigExtract(4, -1, 0, false));
			HoughTransformParameters hough = new HoughParametersPolar(0.5,180);
			HoughTransformBinary alg = new HoughTransformBinary(extractor,hough);
			alg.setMaxLines(1);
			return alg;
		}
	}

	@Nested
	class Binary_MT extends CommonHoughBinaryChecks {

		@Override
		HoughTransformBinary createAlgorithm() {
			NonMaxSuppression extractor = FactoryFeatureExtractor.nonmax(new ConfigExtract(4, -1, 0, false));
			HoughTransformParameters hough = new HoughParametersPolar(0.5,180);
			HoughTransformBinary alg = new HoughTransformBinary_MT(extractor,hough);
			alg.setMaxLines(1);
			return alg;
		}
	}
}
