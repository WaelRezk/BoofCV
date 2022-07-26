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

package boofcv.core.image.border;

import boofcv.struct.border.BorderIndex1D;

/**
 * Returns the closest point inside the image based on Manhattan distance.
 *
 * @author Peter Abeles
 */
public class BorderIndex1D_Extend extends BorderIndex1D {
	@Override
	public int getIndex( int index ) {
		if (index < 0) {
			return 0;
		} else if (index >= length)
			return length - 1;
		return index;
	}

	@Override
	public BorderIndex1D_Extend copy() {
		return new BorderIndex1D_Extend();
	}
}
