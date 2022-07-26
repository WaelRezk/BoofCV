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

package boofcv.struct.feature;

import java.util.Arrays;

/**
 * Feature description storage in an array of unsigned bytes.
 *
 * @author Peter Abeles
 */
public class TupleDesc_U8 extends TupleDesc_I8<TupleDesc_U8> {

	public TupleDesc_U8( int numFeatures ) {
		super(numFeatures);
	}

	public TupleDesc_U8( byte... values ) {
		super(values.length);
		System.arraycopy(values, 0, this.data, 0, values.length);
	}

	public int get( int index ) {return this.data[index] & 0xFF;}

	@Override public void setTo( byte... value ) {
		System.arraycopy(value, 0, this.data, 0, this.data.length);
	}

	public void fill( int value ) {
		Arrays.fill(this.data, (byte)value);
	}

	@Override public double getDouble( int index ) {
		return data[index] & 0xFF;
	}

	@Override public TupleDesc_U8 newInstance() {
		return new TupleDesc_U8(data.length);
	}
}
