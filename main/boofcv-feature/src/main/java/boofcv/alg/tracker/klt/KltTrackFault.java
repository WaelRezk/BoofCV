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

package boofcv.alg.tracker.klt;

/**
 * Different types of faults that can cause a KLT track to be dropped.
 *
 * @author Peter Abeles
 */
public enum KltTrackFault {
	/**
	 * The feature was successfully tracked
	 */
	SUCCESS,
	/**
	 * The feature has moved farther than it could possibly be tracked
	 */
	DRIFTED,
	/**
	 * The tracked move out of the image bounds.
	 */
	OUT_OF_BOUNDS,
	/**
	 * Miscellaneous track failure
	 */
	FAILED,
	/**
	 * The feature's error was too large
	 */
	LARGE_ERROR
}
