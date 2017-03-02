package testing;

import activationFunction.ActivationFunction;
import activationFunction.ActivationGaussian;
import activationFunction.ActivationLOG;
import activationFunction.ActivationLinear;
import activationFunction.ActivationSIN;
import activationFunction.ActivationSigmoid;
import activationFunction.ActivationTANH;
import supervisedLearning.BackPropagationNetwork;

/**
 * @author Lauri Turunen
 *
 *	This class can be used to test the neural network in solving a problems with simple input data. 
 */

public class Test {
	
	public static double TEST_INPUT[][] = { {1.1, 1.0 }, { 2.0, 1.0 },
		{ 1.0, 1.0 }, { 0.5, 0.5 } };

	public static double TEST_IDEAL[][] = { { 2.2 }, { 3.0 }, { 2.0 }, { 1.0 } };
	
	public static void main(String[] args) throws Exception 
	{

		int[] hidden = new int[1];
		hidden[0]=5;
		ActivationFunction acFunct=new ActivationSigmoid();
	
		/*
		 * Best for sigmoid function:
		 * LR 0.1, Momentum: 0.1
		 */

		/*
		 * For SIN function: 
		 * 0.00001,0.0001
		 */

		BackPropagationNetwork feedForward1=
		new BackPropagationNetwork(2,hidden,1,acFunct,0.15,0.1,TEST_INPUT,TEST_IDEAL,true);

		feedForward1.setEpoch(1000);
		feedForward1.iterate();
		feedForward1.printBestResult();

		
		
	}

}
