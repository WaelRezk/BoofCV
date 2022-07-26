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

package boofcv.abst.fiducial;

import boofcv.alg.distort.LensDistortionNarrowFOV;
import boofcv.struct.image.ImageBase;
import boofcv.struct.image.ImageType;
import georegression.struct.point.Point2D_F64;
import georegression.struct.se.Se3_F64;
import georegression.struct.shapes.Polygon2D_F64;
import org.jetbrains.annotations.Nullable;

/**
 * Interface for detecting fiducial markers and their location in the image. Optionally, some implementations
 * can produce a unique ID for each fiducial and their 3D pose in the world. Implementations of this interface
 * will be detectors only, meaning that they are not trackers, meaning that each call to detect will produce
 * identical results only dependent on the input image and not the past history.
 *
 * @author Peter Abeles
 * @see FiducialTracker
 */
public interface FiducialDetector<T extends ImageBase<T>> {
	/**
	 * Detects fiducials inside the image. Each call to this function only depends upon the input image. The
	 * previous calls do not affect it's outcome.
	 *
	 * @param input Input image. Not modified.
	 */
	void detect( T input );

	/**
	 * The total number of fiducial found
	 *
	 * @return number of targets found
	 */
	int totalFound();

	/**
	 * Returns where in the image the fiducial is. Typically this will be the fiducial's visual center. Note that
	 * the visual center is unlikely to be the projection of the 3D geometric center. To find the former you need
	 * to reproject it using the found fiducialToCamera.
	 *
	 * <p>NOTE: The reprojected center might not be the same as the location returned here.</p>
	 *
	 * @param which Fiducial's index
	 * @param location (output) Storage for the transform. modified.
	 */
	void getCenter( int which, Point2D_F64 location );

	/**
	 * Used to retrieve the bounds around a marker in the image. How the bounds are defined is left up the
	 * implementation. It could be a simple rectangle or it could be corner features.
	 *
	 * @param which Which fiducial.
	 * @param storage (Optional) Storage for fiducials.
	 * @return Found marker. Points are owned by the caller and will not be modified in the future.
	 */
	Polygon2D_F64 getBounds( int which, @Nullable Polygon2D_F64 storage );

	/**
	 * If applicable, returns the ID of the fiducial found. Call {@link #hasID()} to see if this function
	 * returns a valid value.
	 *
	 * @param which Detected fiducial's index
	 * @return ID of the fiducial
	 */
	long getId( int which );

	/**
	 * If applicable, returns a message associated with the fiducial. Call {@link #hasMessage()} ()} to see if this function
	 * returns a valid value.
	 *
	 * @param which Detected fiducial's index
	 * @return Encoded message
	 */
	String getMessage( int which );

	/**
	 * Specifies how to remove lens distortion from the input image and how to convert pixels into normalized
	 * image coordinates.
	 *
	 * @param distortion Lens distortion model. null if you want to remove a lens distortion model that had previously
	 * been set.
	 * @param width Input image's width.
	 * @param height Input image's height
	 */
	void setLensDistortion( @Nullable LensDistortionNarrowFOV distortion, int width, int height );

	/**
	 * Returns the intrinsic parameters that it
	 *
	 * @return intrinsic parameters
	 */
	@Nullable LensDistortionNarrowFOV getLensDistortion();

	/**
	 * <p>Computes metrics which represents the fiducial's pose estimate stability given its current observed state.
	 * This can be viewed as an estimate of the pose estimate's precision, but not its accuracy. These numbers are
	 * generated by perturbing landmarks (by the user provided amount) and seeing how it affects
	 * the pose estimate..</p>
	 *
	 * <p>The metrics should be considered more qualitative than quantitative
	 * since exactly how this metric is computed isn't specified and can vary depending on target type of
	 * implementation type. The results could every vary each time its called, even with the the exact same inputs.
	 * </p>
	 *
	 * @param which Index of which fiducial the stability is being requested from
	 * @param disturbance Amount of the applied disturbance, in pixels. Try 0.25
	 * @param results (output) Storage for stability metrics.
	 * @return true if successful or false if it failed for some reason
	 */
	boolean computeStability( int which, double disturbance, FiducialStability results );

	/**
	 * Used to retrieve the transformation from the fiducial's reference frame to the camera's reference frame.
	 *
	 * @param which Fiducial's index
	 * @param fiducialToCamera (output) Storage for the transform. modified.
	 * @return true if could estimate the location or false if it couldn't
	 */
	boolean getFiducialToCamera( int which, Se3_F64 fiducialToCamera );

	/**
	 * Returns the width of the fiducial in world units. If not square then it returns a reasonable
	 * approximation. Intended for use in visualization and not precise calculations.
	 *
	 * @param which Fiducial's index
	 * @return Fiducial's width.
	 */
	double getWidth( int which );

	/**
	 * If true then 3D information is available for the fiducial. In general a len distortion model must be
	 * provided by invoking {@link #setLensDistortion(LensDistortionNarrowFOV, int, int)}. The following functions are then
	 * enabled:
	 * <ol>
	 *     <li>{@link #computeStability}</li>
	 *     <li>{@link #getFiducialToCamera}</li>
	 * </ol>
	 */
	boolean is3D();

	/**
	 * If true then {@link #getId(int)} returns a valid unique number
	 *
	 * @return boolean
	 */
	boolean hasID();

	/**
	 * If true then {@link #getMessage(int)} returns a valid message
	 *
	 * @return boolean
	 */
	boolean hasMessage();

	/**
	 * Type of input image
	 */
	ImageType<T> getInputType();
}
