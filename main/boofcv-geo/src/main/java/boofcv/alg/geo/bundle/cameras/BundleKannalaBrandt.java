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

package boofcv.alg.geo.bundle.cameras;

import boofcv.abst.geo.bundle.BundleAdjustmentCamera;
import boofcv.misc.BoofMiscOps;
import boofcv.struct.calib.CameraKannalaBrandt;
import georegression.geometry.UtilPoint3D_F64;
import georegression.struct.point.Point2D_F64;
import lombok.Getter;
import org.jetbrains.annotations.Nullable;

import static boofcv.alg.distort.kanbra.KannalaBrandtUtils_F64.*;

/**
 * Implementation of {@link CameraKannalaBrandt} for bundle adjustment
 *
 * @author Peter Abeles
 */
public class BundleKannalaBrandt implements BundleAdjustmentCamera {
	/** Camera model parameters */
	@Getter public final CameraKannalaBrandt model = new CameraKannalaBrandt();

	/** forces skew to be zero */
	@Getter public boolean zeroSkew;

	/** True if a there are asymmetric distortion terms */
	@Getter public boolean isAsymmetric;

	// Number of degrees of freedom in the model
	int dof;

	// Storage for asymmetric distortion polynomial results
	double polyRad, polyRadTrig; // model.radial and model.radialTrig
	double polyTan, polyTanTrig; // model.tangent and model.tangentTrig

	// Gradient of polyTrig function generated by differentiated by coefficients
	double[] polyTrigGradient = new double[4];

	public BundleKannalaBrandt( CameraKannalaBrandt model ) {
		configure(model.skew == 0.0, model.symmetric.length, model.radial.length);
		this.model.setTo(model);
	}

	public BundleKannalaBrandt() {}

	/**
	 * Configures the parameters in the model
	 *
	 * @param zeroSkew If true, then skew will always be zero
	 * @param numSymmetric Number of symmetric coefficients
	 * @param numAsymmetric Number of asymmetric coefficients.
	 */
	public void configure( boolean zeroSkew, int numSymmetric, int numAsymmetric ) {
		this.zeroSkew = zeroSkew;
		model.configureCoefficients(numSymmetric, numAsymmetric);

		dof = zeroSkew ? 4 : 5;
		dof += model.symmetric.length;
		dof += model.radial.length;
		dof += model.radialTrig.length;
		dof += model.tangent.length;
		dof += model.tangentTrig.length;

		isAsymmetric = numAsymmetric > 0;
	}

	@Override public void setIntrinsic( double[] parameters, int offset ) {
		int index = offset;
		model.fx = parameters[index++];
		model.fy = parameters[index++];
		model.cx = parameters[index++];
		model.cy = parameters[index++];
		if (!zeroSkew)
			model.skew = parameters[index++];

		if (model.symmetric.length > 0) {
			System.arraycopy(parameters, index, model.symmetric, 0, model.symmetric.length);
			index += model.symmetric.length;
		}

		if (model.radial.length > 0) {
			System.arraycopy(parameters, index, model.radial, 0, model.radial.length);
			index += model.radial.length;
			System.arraycopy(parameters, index, model.radialTrig, 0, model.radialTrig.length);
			index += model.radialTrig.length;
		}

		if (model.tangent.length > 0) {
			System.arraycopy(parameters, index, model.tangent, 0, model.tangent.length);
			index += model.tangent.length;
			System.arraycopy(parameters, index, model.tangentTrig, 0, model.tangentTrig.length);
		}
	}

	@Override public void getIntrinsic( double[] parameters, int offset ) {
		BoofMiscOps.checkTrue(parameters.length >= offset + dof);
		int index = offset;

		parameters[index++] = model.fx;
		parameters[index++] = model.fy;
		parameters[index++] = model.cx;
		parameters[index++] = model.cy;
		if (!zeroSkew)
			parameters[index++] = model.skew;

		if (model.symmetric.length > 0) {
			System.arraycopy(model.symmetric, 0, parameters, index, model.symmetric.length);
			index += model.symmetric.length;
		}

		if (model.radial.length > 0) {
			System.arraycopy(model.radial, 0, parameters, index, model.radial.length);
			index += model.radial.length;
			System.arraycopy(model.radialTrig, 0, parameters, index, model.radialTrig.length);
			index += model.radialTrig.length;
		}

		if (model.tangent.length > 0) {
			System.arraycopy(model.tangent, 0, parameters, index, model.tangent.length);
			index += model.tangent.length;
			System.arraycopy(model.tangentTrig, 0, parameters, index, model.tangentTrig.length);
		}
	}

	@Override public void project( double camX, double camY, double camZ, Point2D_F64 output ) {
		// angle between incoming ray and principle axis
		//    Principle Axis = (0,0,z)
		//    Incoming Ray   = (x,y,z)
		double theta = Math.acos(camZ/UtilPoint3D_F64.norm(camX, camY, camZ)); // uses dot product

		// compute symmetric projection function
		double r = polynomial(model.symmetric, theta);

		// angle on the image plane of the incoming ray
		double phi = Math.atan2(camY, camX);
		double cosphi = Math.cos(phi); // u_r[0] or u_phi[1]
		double sinphi = Math.sin(phi); // u_r[1] or -u_phi[0]

		// distorted (normalized) coordinates
		double distX, distY;
		if (isAsymmetric) {
			// distortion terms. radial and tangential
			double disRad = polynomial(model.radial, theta)*polytrig(model.radialTrig, cosphi, sinphi);
			double disTan = polynomial(model.tangent, theta)*polytrig(model.tangentTrig, cosphi, sinphi);

			// put it all together to get normalized image coordinates
			distX = (r + disRad)*cosphi - disTan*sinphi;
			distY = (r + disRad)*sinphi + disTan*cosphi;
		} else {
			distX = r*cosphi;
			distY = r*sinphi;
		}

		// project into pixels
		double skew = zeroSkew ? 0.0 : model.skew;
		output.x = (model.fx*distX + skew*distY + model.cx);
		output.y = (model.fy*distY + model.cy);
	}

	@Override
	public void jacobian( final double camX, final double camY, final double camZ,
						  double[] pointX, double[] pointY,
						  boolean computeIntrinsic, @Nullable double[] calibX, @Nullable double[] calibY ) {

		// Compute intermediate variables used in forward projection model
		double norm2 = camX*camX + camY*camY + camZ*camZ;
		double norm = Math.sqrt(norm2);
		double norm3 = norm2*norm;

		double acos_input = camZ/norm;
		double theta = Math.acos(acos_input);
		double r = polynomial(model.symmetric, theta);
		double phi = Math.atan2(camY, camX);
		double cosphi = Math.cos(phi);
		double sinphi = Math.sin(phi);

		// Compute individual partials then apply chain rule. Roughly ordered the same as the original forward equations
		double theta_d = -1.0/Math.sqrt(1.0 - acos_input*acos_input);
		double theta_dX = -theta_d*camZ*camX/norm3;
		double theta_dY = -theta_d*camZ*camY/norm3;
		double theta_dZ = theta_d*(1.0/norm - camZ*camZ/norm3);

		double r_dTheta = polynomialDerivative(model.symmetric, theta);

		double phi_dX = -camY/(camX*camX + camY*camY);
		double phi_dY = camX/(camX*camX + camY*camY);

		// gradient of distorted coordinate
		double distX_dX = r_dTheta*theta_dX*cosphi - r*sinphi*phi_dX;
		double distX_dY = r_dTheta*theta_dY*cosphi - r*sinphi*phi_dY;
		double distX_dZ = r_dTheta*theta_dZ*cosphi;

		double distY_dX = r_dTheta*theta_dX*sinphi + r*cosphi*phi_dX;
		double distY_dY = r_dTheta*theta_dY*sinphi + r*cosphi*phi_dY;
		double distY_dZ = r_dTheta*theta_dZ*sinphi;

		// Distorted coordinate with only symmetric terms
		double distX = r*cosphi;
		double distY = r*sinphi;

		// Add asymmetric component
		if (isAsymmetric) {
			double polyRad_dTheta = polynomialDerivative(model.radial, theta);
			double polyRadTrig_dPhi = polytrigDerivative(model.radialTrig, cosphi, sinphi);
			double polyTan_dTheta = polynomialDerivative(model.tangent, theta);
			double polyTanTrig_dPhi = polytrigDerivative(model.tangentTrig, cosphi, sinphi);

			polyRad = polynomial(model.radial, theta);
			polyRadTrig = polytrig(model.radialTrig, cosphi, sinphi);

			polyTan = polynomial(model.tangent, theta);
			polyTanTrig = polytrig(model.tangentTrig, cosphi, sinphi);

			double disRad_dX = polyRad_dTheta*theta_dX*polyRadTrig + polyRad*polyRadTrig_dPhi*phi_dX;
			double disRad_dY = polyRad_dTheta*theta_dY*polyRadTrig + polyRad*polyRadTrig_dPhi*phi_dY;
			double disRad_dZ = polyRad_dTheta*theta_dZ*polyRadTrig;

			double disTan_dX = polyTan_dTheta*theta_dX*polyTanTrig + polyTan*polyTanTrig_dPhi*phi_dX;
			double disTan_dY = polyTan_dTheta*theta_dY*polyTanTrig + polyTan*polyTanTrig_dPhi*phi_dY;
			double disTan_dZ = polyTan_dTheta*theta_dZ*polyTanTrig;

			// gradient of distRad*cos(phi)
			distX_dX += disRad_dX*cosphi - polyRad*polyRadTrig*sinphi*phi_dX;
			distX_dY += disRad_dY*cosphi - polyRad*polyRadTrig*sinphi*phi_dY;
			distX_dZ += disRad_dZ*cosphi;

			// gradient of -distTan*sin(phi)
			distX_dX -= disTan_dX*sinphi + polyTan*polyTanTrig*cosphi*phi_dX;
			distX_dY -= disTan_dY*sinphi + polyTan*polyTanTrig*cosphi*phi_dY;
			distX_dZ -= disTan_dZ*sinphi;

			// gradient of distRad*sin(phi)
			distY_dX += disRad_dX*sinphi + polyRad*polyRadTrig*cosphi*phi_dX;
			distY_dY += disRad_dY*sinphi + polyRad*polyRadTrig*cosphi*phi_dY;
			distY_dZ += disRad_dZ*sinphi;

			// gradient of distTan*cos(phi)
			distY_dX += disTan_dX*cosphi - polyTan*polyTanTrig*sinphi*phi_dX;
			distY_dY += disTan_dY*cosphi - polyTan*polyTanTrig*sinphi*phi_dY;
			distY_dZ += disTan_dZ*cosphi;

			// Compute the distorted coordinate. Needed below.
			distX += polyRad*polyRadTrig*cosphi - polyTan*polyTanTrig*sinphi;
			distY += polyRad*polyRadTrig*sinphi + polyTan*polyTanTrig*cosphi;
		}

		pointX[0] = model.fx*distX_dX;
		pointX[1] = model.fx*distX_dY;
		pointX[2] = model.fx*distX_dZ;

		pointY[0] = model.fy*distY_dX;
		pointY[1] = model.fy*distY_dY;
		pointY[2] = model.fy*distY_dZ;

		if (!zeroSkew) {
			pointX[0] += model.skew*distY_dX;
			pointX[1] += model.skew*distY_dY;
			pointX[2] += model.skew*distY_dZ;
		}

		if (!computeIntrinsic || calibX == null || calibY == null)
			return;

		// @formatter:off
		// Intrinsic parameters K
		calibX[0] = distX;  calibY[0] = 0;
		calibX[1] = 0;      calibY[1] = distY;
		calibX[2] = 1;      calibY[2] = 0;
		calibX[3] = 0;      calibY[3] = 1;

		int index;
		if (!zeroSkew) {
			calibX[4] = distY;  calibY[4] = 0;
			index = 5;
		} else {
			index = 4;
		}
		// @formatter:on

		// Symmetric distortion
		double skew = zeroSkew ? 0.0 : model.skew;
		double pows = theta;
		for (int i = 0; i < model.symmetric.length; i++, index++) {
			double r_d = pows;
			calibX[index] = model.fx*r_d*cosphi + skew*r_d*sinphi;
			calibY[index] = model.fy*r_d*sinphi;
			pows *= theta*theta;
		}

		// Asymmetric distortion
		if (!isAsymmetric)
			return;

		polytrigGradient(cosphi, sinphi, polyTrigGradient);

		double powr = theta;
		for (int i = 0; i < model.radial.length; i++, index++) {
			double disRad_d = powr*polyRadTrig;
			calibX[index] = model.fx*disRad_d*cosphi + skew*disRad_d*sinphi;
			calibY[index] = model.fy*disRad_d*sinphi;
			powr *= theta*theta;
		}

		for (int i = 0; i < polyTrigGradient.length; i++, index++) {
			double disRad_d = polyRad*polyTrigGradient[i];
			calibX[index] = model.fx*disRad_d*cosphi + skew*disRad_d*sinphi;
			calibY[index] = model.fy*disRad_d*sinphi;
		}

		double powt = theta;
		for (int i = 0; i < model.tangent.length; i++, index++) {
			double disTan_d = powt*polyTanTrig;
			calibX[index] = -model.fx*disTan_d*sinphi + skew*disTan_d*cosphi;
			calibY[index] = model.fy*disTan_d*cosphi;
			powt *= theta*theta;
		}

		for (int i = 0; i < polyTrigGradient.length; i++, index++) {
			double disTan_d = polyTan*polyTrigGradient[i];
			calibX[index] = -model.fx*disTan_d*sinphi + skew*disTan_d*cosphi;
			calibY[index] = model.fy*disTan_d*cosphi;
		}
	}

	@Override public int getIntrinsicCount() {
		return dof;
	}
}
