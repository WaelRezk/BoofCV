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

package boofcv.struct.sfm;

import georegression.struct.point.Point2D_F64;
import georegression.struct.point.Point3D_F64;

/**
 * Stereo observations composed on a 3D location and two image observations.
 *
 * @author Peter Abeles
 */
public class Stereo2D3D {

	// observation of point in left and right camera
	// normalized image coordinates
	public Point2D_F64 leftObs = new Point2D_F64();
	public Point2D_F64 rightObs = new Point2D_F64();

	// 3D coordinate of the point
	public Point3D_F64 location = new Point3D_F64();

	public Stereo2D3D( Point2D_F64 leftObs, Point2D_F64 rightObs, Point3D_F64 location ) {
		this.leftObs = leftObs;
		this.rightObs = rightObs;
		this.location = location;
	}

	public Stereo2D3D() {}
}
