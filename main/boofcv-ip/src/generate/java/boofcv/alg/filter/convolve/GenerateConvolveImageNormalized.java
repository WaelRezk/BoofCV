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

package boofcv.alg.filter.convolve;

import boofcv.generate.AutoTypeImage;
import boofcv.generate.CodeGeneratorBase;

import java.io.FileNotFoundException;

/**
 * NOTE: There is a tinny bit of manual work required. need to comment out a few lines to unroll
 *
 * @author Peter Abeles
 */
public class GenerateConvolveImageNormalized extends CodeGeneratorBase {

	boolean isInteger;
	String kernelType;
	String borderName;
	String inputName, outputName, typeIn, typeOut, sumType;
	String borderType;

	int totalFunctions = 0;

	@Override
	public void generateCode() {
		printPreamble();
		printAllOps(AutoTypeImage.F32, AutoTypeImage.F32);
		printAllOps(AutoTypeImage.F64, AutoTypeImage.F64);
		printAllOps(AutoTypeImage.U8,  AutoTypeImage.I8);
		printAllOps(AutoTypeImage.S16, AutoTypeImage.I16);
		printAllOps(AutoTypeImage.U16, AutoTypeImage.I16);
		printAllOps(AutoTypeImage.S32, AutoTypeImage.S32);

		out.println("}");

		System.out.println("Total functions generated "+totalFunctions);
	}

	private void printPreamble() {
		out.print(
				"import boofcv.alg.filter.convolve.border.ConvolveJustBorder_General_SB;\n" +
				"import boofcv.struct.convolve.*;\n" +
				"import boofcv.struct.border.*;\n" +
				"import boofcv.struct.image.*;\n" +
				"import boofcv.alg.filter.convolve.normalized.*;\n" +
				"import boofcv.alg.filter.kernel.KernelMath;\n");
		out.println();
		out.print("/**\n" +
				" * <p>\n" +
				" * Convolves a kernel across an image and scales the kernel such that the sum of the portion inside\n" +
				" * the image sums up to one.\n" +
				" * </p>\n" +
				" * <p>Automatically generated by "+getClass().getSimpleName()+". DO NOT MODIFY</p>\n" +
				" *\n" +
				" * @author Peter Abeles\n" +
				" */\n" +
				"@SuppressWarnings({\"ForLoopReplaceableByForEach\", \"unchecked\"})\n" +
				"public class "+className+" {\n\n");
	}

	private void printAllOps(AutoTypeImage input, AutoTypeImage output)
	{
		kernelType = input.getKernelType();
		borderType = input.getBorderNameSB();
		typeIn = input.name();
		typeOut = output.name();
		sumType = input.getSumType();
		isInteger = input.isInteger();

		inputName = input.getSingleBandName();
		outputName = output.getSingleBandName();
		borderName = "ImageBorder_";

		for( boolean border : new boolean[]{false,true} ){
			printFunction("horizontal", true, border);
			printFunction("vertical", true, border);
			printFunction("convolve", true, border);
		}

		inputName = input.getInterleavedName();
		outputName = output.getInterleavedName();
		borderName = "ImageBorder_IL_";
		printFunction("horizontal", false,false);
		printFunction("vertical", false,false);
		printFunction("convolve", false,false);
	}

	private void printFunction(  String name , boolean singleBand , boolean border ) {

		totalFunctions++;

		String dimen = name.equals("convolve") ? "2D" : "1D";
		String docName = name.equals("convolve") ? "" : " "+name;

		String suffice = singleBand ? "SB" : "IL";
//		String suffice2 = singleBand ? "" : "B";

		String kernelTypeName = "Kernel"+dimen+"_"+kernelType;

		String overrideName = name.equals("convolve") ? "Convolve" : name.equals("horizontal") ? "Horizontal" : "Vertical";
		String workArray = (name.equals("convolve")||(name.equals("vertical") && isInteger)) && singleBand ? ", null" : "";

		String borderArgument = border ? borderType+" bsrc" : "";

		String commentsOverride = border ? "//" : "";

		String justBorder;
		if( border ) {
			if( isInteger )
				justBorder = "\t\t\tConvolveJustBorder_General_SB."+name+"(kernel, bsrc, dst, kernel.computeSum());\n";
			else
				justBorder = "\t\t\tConvolveJustBorder_General_SB."+name+"(kernel, bsrc, dst);\n";
		} else {
			justBorder = "\t\t\tConvolveNormalized_JustBorder_"+suffice+"."+name+"(kernel, src, dst);\n";
		}
		String bsrc = border ? ", bsrc" : "";

		out.print(
				"\t/**\n" +
				"\t * Performs a"+docName+" "+dimen+" normalized convolution across the image.\n" +
				"\t *\n" +
				"\t * @param src The original image. Not modified.\n" +
				"\t * @param dst Where the resulting image is written to. Modified.\n" +
				"\t * @param kernel The kernel that is being convolved. Not modified.\n" +
				"\t */\n" );

		out.print(functionSignature(1,"public static void",name,kernelTypeName+" kernel",inputName+" src",outputName+" dst", borderArgument));
		out.print("\t\tdst.reshape(src.width,src.height);\n" +
				"\n" +
				"\t\t"+commentsOverride+"if( BOverrideConvolveImageNormalized.invokeNative"+overrideName+"(kernel,src,dst) )\n"+
				"\t\t"+commentsOverride+"\treturn;\n\n");

		if( border ) {
			out.print("\t\tbsrc.setImage(src);\n\n");
		}

		String insideTest;
		if( name.equals("horizontal") ) {
			insideTest = "kernel.width >= src.width";
		} else if( name.equals("vertical") ) {
			insideTest = "kernel.width >= src.height";
		} else {
			insideTest = "kernel.width >= src.width || kernel.width >= src.height";
		}

		if( isInteger ) {
			out.print("\t\tif( "+insideTest+" ) {\n" +
					"\t\t\tConvolveNormalizedNaive_"+suffice+"."+name+"(kernel, src, dst"+bsrc+");\n" +
					"\t\t} else {\n" +
					"\t\t\tConvolveImageNoBorder."+name+"(kernel, src, dst, kernel.computeSum()"+workArray+");\n" +
					justBorder +
					"\t\t}\n");
		} else {
			out.print("\t\tif( "+insideTest+" ) {\n" +
					"\t\t\tConvolveNormalizedNaive_"+suffice+"."+name+"(kernel, src, dst"+bsrc+");\n" +
					"\t\t} else {\n" +
					"\t\t\tif( Math.abs(kernel.computeSum() - 1.0f) > 1e-4f ) {\n" +
					"\t\t\t\t"+kernelTypeName+" k = kernel.copy();\n" +
					"\t\t\t\tKernelMath.normalizeSumToOne(k);\n" +
					"\t\t\t\tkernel = k;\n" +
					"\t\t\t}\n" +
					"\t\t\tConvolveImageNoBorder."+name+"(kernel,src,dst);\n" +
					justBorder +
					"\t\t}\n");
		}
		out.print("\t}\n\n");
	}

	public static void main(String[] args) throws FileNotFoundException {
		var gen = new GenerateConvolveImageNormalized();
		gen.setModuleName("boofcv-ip");
		gen.parseArguments(args);
		gen.generate();
	}
}