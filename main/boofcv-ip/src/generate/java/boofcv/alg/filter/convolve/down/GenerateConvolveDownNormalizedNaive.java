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

package boofcv.alg.filter.convolve.down;

import boofcv.generate.CodeGeneratorBase;

import java.io.FileNotFoundException;

/**
 * Code generator which creates re-normalizing convolution code
 *
 * @author Peter Abeles
 */
public class GenerateConvolveDownNormalizedNaive extends CodeGeneratorBase {
	String divide;

	@Override
	public void generateCode() {
		printPreamble();
		printAllOps("F32", "GrayF32","GrayF32","float","float");
		printAllOps("S32", "GrayU8","GrayI8","int","int");
		printAllOps("S32", "GrayS16","GrayI16","int","int");
		out.println("}");
	}

	private void printPreamble() {
		out.print("import boofcv.struct.convolve.Kernel1D_F32;\n" +
				"import boofcv.struct.convolve.Kernel1D_S32;\n" +
				"import boofcv.struct.convolve.Kernel2D_F32;\n" +
				"import boofcv.struct.convolve.Kernel2D_S32;\n" +
				"import boofcv.struct.image.*;\n" +
				"\n" +
				"import javax.annotation.Generated;\n" +
				"\n" +
				"/**\n" +
				" * Down convolution with kernel renormalization around image borders. Unoptimized naive implementation.\n" +
				generateDocString("Peter Abeles") +
				"public class "+className+" {\n\n");
	}

	private void printAllOps( String kernelType , String inputType , String outputType ,
							  String kernelData, String sumType) {
		boolean isInteger = kernelType.compareTo("S32") == 0;
		divide = isInteger ? "(total + div/2)/div" : "total/div";

		printHorizontal(kernelType,inputType,outputType,kernelData,sumType);
		printVertical(kernelType,inputType,outputType,kernelData,sumType);
		printConvolve(kernelType,inputType,outputType,kernelData,sumType);
	}

	private void printHorizontal( String kernelType , String inputType , String outputType ,
								  String kernelData, String sumType ) {
		out.print("\tpublic static void horizontal( Kernel1D_"+kernelType+" kernel, "+inputType+" input, "+outputType+" output, int skip ) {\n" +
				"\t\toutput.reshape(input.width/skip, input.height);\n" +
				"\n" +
				"\t\tfinal int radius = kernel.getRadius();\n" +
				"\n" +
				"\t\tfinal int width = input.width - input.width%skip;\n" +
				"\t\tfinal int height = input.height;\n" +
				"\n" +
				"\t\tfor (int y = 0; y < height; y++) {\n" +
				"\t\t\tfor (int x = 0; x < width; x += skip) {\n" +
				"\t\t\t\t"+sumType+" total = 0;\n" +
				"\t\t\t\t"+sumType+" div = 0;\n" +
				"\n" +
				"\t\t\t\tint startX = x - radius;\n" +
				"\t\t\t\tint endX = x + radius;\n" +
				"\n" +
				"\t\t\t\tif (startX < 0) startX = 0;\n" +
				"\t\t\t\tif (endX >= input.width) endX = input.width - 1;\n" +
				"\n" +
				"\t\t\t\tfor (int j = startX; j <= endX; j++) {\n" +
				"\t\t\t\t\t"+kernelData+" v = kernel.get(j - x + radius);\n" +
				"\t\t\t\t\ttotal += input.get(j, y)*v;\n" +
				"\t\t\t\t\tdiv += v;\n" +
				"\t\t\t\t}\n" +
				"\t\t\t\toutput.set(x/skip, y, "+divide+");\n" +
				"\t\t\t}\n" +
				"\t\t}\n" +
				"\t}\n\n");
	}

	private void printVertical( String kernelType , String inputType , String outputType ,
								  String kernelData, String sumType ) {

		out.print("\tpublic static void vertical( Kernel1D_"+kernelType+" kernel, "+inputType+" input, "+outputType+" output, int skip ) {\n" +
				"\t\toutput.reshape(input.width, input.height/skip);\n" +
				"\n" +
				"\t\tfinal int radius = kernel.getRadius();\n" +
				"\n" +
				"\t\tfinal int width = input.width;\n" +
				"\t\tfinal int height = input.height - input.height%skip;\n" +
				"\n" +
				"\t\tfor (int y = 0; y < height; y += skip) {\n" +
				"\t\t\tfor (int x = 0; x < width; x++) {\n" +
				"\t\t\t\t"+sumType+" total = 0;\n" +
				"\t\t\t\t"+sumType+" div = 0;\n" +
				"\n" +
				"\t\t\t\tint startY = y - radius;\n" +
				"\t\t\t\tint endY = y + radius;\n" +
				"\n" +
				"\t\t\t\tif (startY < 0) startY = 0;\n" +
				"\t\t\t\tif (endY >= input.height) endY = input.height - 1;\n" +
				"\n" +
				"\t\t\t\tfor (int i = startY; i <= endY; i++) {\n" +
				"\t\t\t\t\t"+kernelData+" v = kernel.get(i - y + radius);\n" +
				"\t\t\t\t\ttotal += input.get(x, i)*v;\n" +
				"\t\t\t\t\tdiv += v;\n" +
				"\t\t\t\t}\n" +
				"\t\t\t\toutput.set(x, y/skip, "+divide+" );\n" +
				"\t\t\t}\n" +
				"\t\t}\n" +
				"\t}\n\n");
	}

	private void printConvolve( String kernelType , String inputType , String outputType ,
								  String kernelData, String sumType ) {

		out.print("\tpublic static void convolve( Kernel2D_"+kernelType+" kernel, "+inputType+" input, "+outputType+" output, int skip ) {\n" +
				"\t\toutput.reshape(input.width/skip, input.height/skip);\n" +
				"\n" +
				"\t\tfinal int radius = kernel.getRadius();\n" +
				"\n" +
				"\t\tfinal int width = input.width - input.width%skip;\n" +
				"\t\tfinal int height = input.height - input.height%skip;\n" +
				"\n" +
				"\t\tfor (int y = 0; y < height; y += skip) {\n" +
				"\t\t\tfor (int x = 0; x < width; x += skip) {\n" +
				"\n" +
				"\t\t\t\tint startX = x - radius;\n" +
				"\t\t\t\tint endX = x + radius;\n" +
				"\n" +
				"\t\t\t\tif (startX < 0) startX = 0;\n" +
				"\t\t\t\tif (endX >= input.width) endX = input.width - 1;\n" +
				"\n" +
				"\t\t\t\tint startY = y - radius;\n" +
				"\t\t\t\tint endY = y + radius;\n" +
				"\n" +
				"\t\t\t\tif (startY < 0) startY = 0;\n" +
				"\t\t\t\tif (endY >= input.height) endY = input.height - 1;\n" +
				"\n" +
				"\t\t\t\t"+sumType+" total = 0;\n" +
				"\t\t\t\t"+sumType+" div = 0;\n" +
				"\n" +
				"\t\t\t\tfor (int i = startY; i <= endY; i++) {\n" +
				"\t\t\t\t\tfor (int j = startX; j <= endX; j++) {\n" +
				"\t\t\t\t\t\t"+kernelData+" v = kernel.get(j - x + radius, i - y + radius);\n" +
				"\t\t\t\t\t\ttotal += input.get(j, i)*v;\n" +
				"\t\t\t\t\t\tdiv += v;\n" +
				"\t\t\t\t\t}\n" +
				"\t\t\t\t}\n" +
				"\t\t\t\toutput.set(x/skip, y/skip, "+divide+" );\n" +
				"\t\t\t}\n" +
				"\t\t}\n" +
				"\t}\n\n");
	}

	public static void main(String[] args) throws FileNotFoundException {
		final var gen = new GenerateConvolveDownNormalizedNaive();
		gen.setModuleName("boofcv-ip");
		gen.generate();
	}
}
