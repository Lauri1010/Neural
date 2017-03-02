/*
 * Encog Artificial Intelligence Framework v2.x
 * Java Version
 * http://www.heatonresearch.com/encog/
 * http://code.google.com/p/encog-java/
 * 
 * Copyright 2008-2009, Heaton Research Inc., and individual contributors.
 * See the copyright.txt in the distribution for a full listing of 
 * individual contributors.
 *
 * This is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of
 * the License, or (at your option) any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this software; if not, write to the Free
 * Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
 * 02110-1301 USA, or see the FSF site: http://www.fsf.org.
 */
package activationFunction;

import utils.BoundMath;
import utils.BoundNumbers;


/**
 * The hyperbolic tangent activation function takes the curved shape of the
 * hyperbolic tangent. This activation function produces both positive and
 * negative output. Use this activation function if both negative and positive
 * output is desired.
 * 
 * This implementation does an approximation of the TANH function, using only a
 * single base e exponent. This has a considerable effect on performance, adds
 * only minimal change to the output compared to a standard TANH calculation.
 * 
 * Based on model by the encog project (http://www.heatonresearch.com/encog)
 * edited by: Lauri Turunen. 
 * 
 */
public class ActivationTANH implements ActivationFunction {

	/**
	 * Serial id for this class.
	 */
	private static final long serialVersionUID = 9121998892720207643L;

	/**
	 * Internal activation function that performs the TANH.
	 * 
	 * @param d
	 *            The input value.
	 * @return The output value.
	 */
	public double activationFunction(double d) {
		final double result = (Math.exp(d*2.0)-1.0)/(Math.exp(d*2.0)+1.0);
		return result;
	}


	/**
	 * @return The object cloned;
	 */
	@Override
	public Object clone() {
		return new ActivationTANH();
	}

	/**
	 * @return Return true, TANH has a derivative.
	 */
	public boolean hasDerivative() {
		return true;
	}
	
	/**
	 * The tanh function derivate
	 */
	@Override
	public double derivativeFunction(double d) {
		return( 1.0-Math.pow(activationFunction(d), 2.0) );
	}
	

}
