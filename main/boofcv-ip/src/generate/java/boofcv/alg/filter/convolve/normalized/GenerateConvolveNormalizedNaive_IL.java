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

package boofcv.alg.filter.convolve.normalized;

import boofcv.generate.AutoTypeImage;
import boofcv.generate.CodeGeneratorBase;

import java.io.FileNotFoundException;

/**
 * Code generator which creates re-normalizing convolution code
 *
 * @author Peter Abeles
 */
public class GenerateConvolveNormalizedNaive_IL extends CodeGeneratorBase {
	String divide;

	@Override
	public void generateCode() {
		printPreamble();
		printAllOps(AutoTypeImage.F32, AutoTypeImage.F32);
		printAllOps(AutoTypeImage.F64, AutoTypeImage.F64);
		printAllOps(AutoTypeImage.U8, AutoTypeImage.I8);
		printAllOps(AutoTypeImage.S16, AutoTypeImage.I16);
		printAllOps(AutoTypeImage.U16, AutoTypeImage.I16);
		printAllOps(AutoTypeImage.S32, AutoTypeImage.S32);

		printVertical2Int(AutoTypeImage.U16, AutoTypeImage.I8);
		printVertical2Int(AutoTypeImage.S32, AutoTypeImage.I16);

		out.println("}");
	}

	private void printPreamble() {
		out.print("import boofcv.struct.convolve.*;\n" +
				"import boofcv.struct.image.*;\n" +
				"import javax.annotation.Generated;\n" +
				"import java.util.Arrays;\n" +
				"\n" +
				"/**\n" +
				" * <p>\n" +
				" * Convolution with kernel renormalization around image borders. Unoptimized naive implementation.\n" +
				" * </p>\n" +
				" * \n" +
				" * <p>\n" +
				" * NOTE: Do not modify. Automatically generated by " + getClass().getSimpleName() + ".\n" +
				" * </p>\n" +
				" * @author Peter Abeles\n" +
				" */\n" +
				"@Generated({\"" + getClass().getCanonicalName() + "\"})\n" +
				"public class " + className + " {\n\n");
	}

	private void printAllOps( AutoTypeImage input, AutoTypeImage output ) {
		boolean isInteger = input.isInteger();
		divide = isInteger ? "= (total[band]+weight/2)/weight" : "/= weight";

		String kernelType = input.getKernelType();
		String inputType = input.getInterleavedName();
		String outputType = output.getInterleavedName();
		String kernelData = input.getSumType();
		String sumType = input.getSumType();

		printHorizontal(kernelType, inputType, outputType, kernelData, sumType);
		printVertical(kernelType, inputType, outputType, kernelData, sumType);
		printConvolve(kernelType, inputType, outputType, kernelData, sumType);
	}

	private void printVertical2Int( AutoTypeImage input, AutoTypeImage output ) {
		divide = "= (total[band]+weight/2)/weight";

		String kernelType = input.getKernelType();
		String inputType = input.getInterleavedName();
		String outputType = output.getInterleavedName();
		String sumType = input.getSumType();

		printVertical2(kernelType, inputType, outputType, sumType);
	}

	private void printHorizontal( String kernelType, String inputType, String outputType,
								  String kernelData, String sumType ) {

		out.print("\tpublic static void horizontal(Kernel1D_" + kernelType + " kernel, " + inputType + " input, " + outputType + " output ) {\n" +
				"\n" +
				"\t\tfinal int offset = kernel.getOffset();\n" +
				"\n" +
				"\t\tfinal int width = input.getWidth();\n" +
				"\t\tfinal int height = input.getHeight();\n" +
				"\t\tfinal int numBands = input.getNumBands();\n" +
				"\t\t\n" +
				"\t\tfinal " + sumType + "[] pixel = new " + sumType + "[ numBands ];\n" +
				"\t\tfinal " + sumType + "[] total = new " + sumType + "[ numBands ];\n" +
				"\n" +
				"\t\tfor (int y = 0; y < height; y++) {\n" +
				"\t\t\tfor( int x = 0; x < width; x++ ) {\n" +
				"\t\t\t\tArrays.fill(total,0);\n" +
				"\t\t\t\t" + sumType + " weight = 0;\n" +
				"\n" +
				"\t\t\t\tint startX = x - offset;\n" +
				"\t\t\t\tint endX = startX+kernel.getWidth();\n" +
				"\n" +
				"\t\t\t\tif( startX < 0 ) startX = 0;\n" +
				"\t\t\t\tif( endX > width ) endX = width;\n" +
				"\n" +
				"\t\t\t\tfor( int j = startX; j < endX; j++ ) {\n" +
				"\t\t\t\t\t" + kernelData + " v = kernel.get(j-x+offset);\n" +
				"\t\t\t\t\tinput.get(j,y, pixel);\n" +
				"\t\t\t\t\tfor (int band = 0; band < numBands; band++) {\n" +
				"\t\t\t\t\t\ttotal[band] += pixel[band]*v;\n" +
				"\t\t\t\t\t}\n" +
				"\t\t\t\t\t\n" +
				"\t\t\t\t\tweight += v;\n" +
				"\t\t\t\t}\n" +
				"\t\t\t\tfor (int band = 0; band < numBands; band++) {\n" +
				"\t\t\t\t\ttotal[band] " + divide + ";\n" +
				"\t\t\t\t}\n" +
				"\t\t\t\toutput.set(x,y, total );\n" +
				"\t\t\t}\n" +
				"\t\t}\n" +
				"\t}\n\n");
	}

	private void printVertical( String kernelType, String inputType, String outputType,
								String kernelData, String sumType ) {

		out.print("\tpublic static void vertical(Kernel1D_" + kernelType + " kernel, " + inputType + " input, " + outputType + " output ) {\n" +
				"\n" +
				"\t\tfinal int offset = kernel.getOffset();\n" +
				"\n" +
				"\t\tfinal int width = input.getWidth();\n" +
				"\t\tfinal int height = input.getHeight();\n" +
				"\t\tfinal int numBands = input.getNumBands();\n" +
				"\n" +
				"\t\tfinal " + sumType + "[] pixel = new " + sumType + "[ numBands ];\n" +
				"\t\tfinal " + sumType + "[] total = new " + sumType + "[ numBands ];\n" +
				"\n" +
				"\t\tfor (int y = 0; y < height; y++) {\n" +
				"\t\t\tfor( int x = 0; x < width; x++ ) {\n" +
				"\t\t\t\tArrays.fill(total,0);\n" +
				"\t\t\t\t" + sumType + " weight = 0;\n" +
				"\n" +
				"\t\t\t\tint startY = y - offset;\n" +
				"\t\t\t\tint endY = startY + kernel.getWidth();\n" +
				"\n" +
				"\t\t\t\tif( startY < 0 ) startY = 0;\n" +
				"\t\t\t\tif( endY > height ) endY = height;\n" +
				"\n" +
				"\t\t\t\tfor( int i = startY; i < endY; i++ ) {\n" +
				"\t\t\t\t\t" + kernelData + " v = kernel.get(i-y+offset);\n" +
				"\t\t\t\t\tinput.get(x,i, pixel);\n" +
				"\t\t\t\t\tfor (int band = 0; band < numBands; band++) {\n" +
				"\t\t\t\t\t\ttotal[band] += pixel[band]*v;\n" +
				"\t\t\t\t\t}\n" +
				"\t\t\t\t\tweight += v;\n" +
				"\t\t\t\t}\n" +
				"\t\t\t\tfor (int band = 0; band < numBands; band++) {\n" +
				"\t\t\t\t\ttotal[band] " + divide + ";\n" +
				"\t\t\t\t}\n" +
				"\t\t\t\toutput.set(x,y, total);\n" +
				"\t\t\t}\n" +
				"\t\t}\n" +
				"\t}\n\n");
	}

	private void printVertical2( String kernelType, String inputType, String outputType, String sumType ) {

		out.print("\tpublic static void vertical(Kernel1D_" + kernelType + " kernelX, Kernel1D_" + kernelType + " kernelY,\n" +
				"\t\t\t\t\t\t\t\t" + inputType + " input, " + outputType + " output ) {\n" +
				"\n" +
				"\t\tfinal int offsetX = kernelX.getOffset();\n" +
				"\t\tfinal int offsetY = kernelY.getOffset();\n" +
				"\n" +
				"\t\tfinal int width = input.getWidth();\n" +
				"\t\tfinal int height = input.getHeight();\n" +
				"\t\tfinal int numBands = input.getNumBands();\n" +
				"\n" +
				"\t\tfinal " + sumType + "[] pixel = new " + sumType + "[ numBands ];\n" +
				"\t\tfinal " + sumType + "[] total = new " + sumType + "[ numBands ];\n" +
				"\n" +
				"\t\tfor (int y = 0; y < height; y++) {\n" +
				"\t\t\tfor (int x = 0; x < width; x++) {\n" +
				"\t\t\t\tArrays.fill(total,0);\n" +
				"\t\t\t\t" + sumType + " weightY = 0;\n" +
				"\n" +
				"\t\t\t\tint startY = y - offsetY;\n" +
				"\t\t\t\tint endY = startY + kernelY.getWidth();\n" +
				"\n" +
				"\t\t\t\tif (startY < 0) startY = 0;\n" +
				"\t\t\t\tif (endY > height) endY = height;\n" +
				"\n" +
				"\t\t\t\tfor (int i = startY; i < endY; i++) {\n" +
				"\t\t\t\t\t" + sumType + " v = kernelY.get(i - y + offsetY);\n" +
				"\t\t\t\t\tinput.get(x, i, pixel);\n" +
				"\t\t\t\t\tfor (int band = 0; band < numBands; band++) {\n" +
				"\t\t\t\t\t\ttotal[band] += pixel[band]*v;\n" +
				"\t\t\t\t\t}\n" +
				"\t\t\t\t\tweightY += v;\n" +
				"\t\t\t\t}\n" +
				"\n" +
				"\t\t\t\tint kerX0 = Math.max(0, offsetX - x);\n" +
				"\t\t\t\tint kerX1 = Math.min(kernelX.getWidth(), width - x + offsetX);\n" +
				"\n" +
				"\t\t\t\t" + sumType + " weightX = 0;\n" +
				"\t\t\t\tfor (int i = kerX0; i < kerX1; i++) {\n" +
				"\t\t\t\t\tweightX += kernelX.get(i);\n" +
				"\t\t\t\t}\n" +
				"\n" +
				"\t\t\t\t" + sumType + " weight = weightX * weightY;\n" +
				"\n" +
				"\t\t\t\tfor (int band = 0; band < numBands; band++) {\n" +
				"\t\t\t\t\ttotal[band] " + divide + ";\n" +
				"\t\t\t\t}\n" +
				"\t\t\t\t\n" +
				"\t\t\t\toutput.set(x,y, total);\n" +
				"\t\t\t}\n" +
				"\t\t}\n" +
				"\t}\n\n");
	}

	private void printConvolve( String kernelType, String inputType, String outputType,
								String kernelData, String sumType ) {

		out.print("\tpublic static void convolve(Kernel2D_" + kernelType + " kernel, " + inputType + " input, " + outputType + " output ) {\n" +
				"\n" +
				"\t\tfinal int offset = kernel.getOffset();\n" +
				"\n" +
				"\t\tfinal int width = input.getWidth();\n" +
				"\t\tfinal int height = input.getHeight();\n" +
				"\t\tfinal int numBands = input.getNumBands();\n" +
				"\n" +
				"\t\tfinal " + sumType + "[] pixel = new " + sumType + "[ numBands ];\n" +
				"\t\tfinal " + sumType + "[] total = new " + sumType + "[ numBands ];\n" +
				"\n" +
				"\t\tfor (int y = 0; y < height; y++) {\n" +
				"\t\t\tfor( int x = 0; x < width; x++ ) {\n" +
				"\n" +
				"\t\t\t\tint startX = x - offset;\n" +
				"\t\t\t\tint endX = startX + kernel.getWidth();\n" +
				"\n" +
				"\t\t\t\tif( startX < 0 ) startX = 0;\n" +
				"\t\t\t\tif( endX > width ) endX = width;\n" +
				"\n" +
				"\t\t\t\tint startY = y - offset;\n" +
				"\t\t\t\tint endY = startY + kernel.getWidth();\n" +
				"\n" +
				"\t\t\t\tif( startY < 0 ) startY = 0;\n" +
				"\t\t\t\tif( endY > height ) endY = height;\n" +
				"\n" +
				"\t\t\t\tArrays.fill(total,0);\n" +
				"\t\t\t\t" + sumType + " weight = 0;\n" +
				"\n" +
				"\t\t\t\tfor( int i = startY; i < endY; i++ ) {\n" +
				"\t\t\t\t\tfor( int j = startX; j < endX; j++ ) {\n" +
				"\t\t\t\t\t\tinput.get(j,i, pixel);\n" +
				"\t\t\t\t\t\t" + kernelData + " v = kernel.get(j-x+offset,i-y+offset);\n" +
				"\t\t\t\t\t\tfor (int band = 0; band < numBands; band++) {\n" +
				"\t\t\t\t\t\t\ttotal[band] += pixel[band]*v;\n" +
				"\t\t\t\t\t\t}\n" +
				"\t\t\t\t\t\tweight += v;\n" +
				"\t\t\t\t\t}\n" +
				"\t\t\t\t}\n" +
				"\t\t\t\tfor (int band = 0; band < numBands; band++) {\n" +
				"\t\t\t\t\ttotal[band] " + divide + ";\n" +
				"\t\t\t\t}\n" +
				"\t\t\t\toutput.set(x,y, total);\n" +
				"\t\t\t}\n" +
				"\t\t}\n" +
				"\t}\n\n");
	}

	public static void main( String[] args ) throws FileNotFoundException {
		GenerateConvolveNormalizedNaive_IL gen = new GenerateConvolveNormalizedNaive_IL();
		gen.generateCode();
	}
}
