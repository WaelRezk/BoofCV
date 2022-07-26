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

package boofcv.alg.feature.detect.interest;

import boofcv.alg.feature.detect.intensity.IntegralImageFeatureIntensity;
import boofcv.alg.feature.detect.intensity.impl.ImplIntegralImageFeatureIntensity;
import boofcv.alg.misc.ImageMiscOps;
import boofcv.alg.transform.ii.IntegralImageOps;
import boofcv.concurrency.BoofConcurrency;
import boofcv.struct.image.GrayF32;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import org.openjdk.jmh.runner.options.TimeValue;

import java.util.Random;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 2)
@Measurement(iterations = 5)
@State(Scope.Benchmark)
@Fork(value = 1)
public class BenchmarkFastHessianFeatureIntensity {
	@Param({"true","false"})
	public boolean concurrent=false;

	public int imageSize=1000;

	static int skip = 1;
	static int size = 15;

	GrayF32 original = new GrayF32(imageSize,imageSize);
	GrayF32 integral = new GrayF32(imageSize,imageSize);
	GrayF32 intensity = new GrayF32(imageSize,imageSize);

	@Setup public void setup() {
		BoofConcurrency.USE_CONCURRENT = concurrent;
		var rand = new Random(234234);

		ImageMiscOps.fillUniform(original,rand,0,200);
		IntegralImageOps.transform(original,integral);
	}

	// @formatter:off
	@Benchmark public void Naive() {ImplIntegralImageFeatureIntensity.hessianNaive(integral,skip,size,intensity);}
	@Benchmark public void Standard() {IntegralImageFeatureIntensity.hessian(integral,skip,size,intensity);}
	// @formatter:on

	public static void main(String[] args) throws RunnerException {
		Options opt = new OptionsBuilder()
				.include(BenchmarkFastHessianFeatureIntensity.class.getSimpleName())
				.warmupTime(TimeValue.seconds(1))
				.measurementTime(TimeValue.seconds(1))
				.build();

		new Runner(opt).run();
	}
}
