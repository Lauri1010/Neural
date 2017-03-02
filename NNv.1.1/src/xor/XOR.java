package xor;

import activationFunction.ActivationFunction;
import activationFunction.ActivationLinear;
import activationFunction.ActivationSIN;
import activationFunction.ActivationSigmoid;
import activationFunction.ActivationTANH;
import supervisedLearning.BackPropagationNetwork;

/**
 * @author Lauri Turunen
 *
 *	This class can be used to test the neural network in solving a XOR problem. 
 */

public class XOR {
	
	public static double XOR_INPUT[][] = { { 0.0, 0.0 }, { 1.0, 0.0 },
		{ 0.0, 1.0 }, { 1.0, 1.0 } };

	public static double XOR_IDEAL[][] = { { 0.0 }, { 1.0 }, { 1.0 }, { 0.0 } };
	
	
	public static void main(String[] args) throws Exception 
	{

		int[] hidden = new int[1];
		hidden[0]=3;
		ActivationFunction acFunct=new ActivationSigmoid();
		
		BackPropagationNetwork feedForward1=
		new BackPropagationNetwork(2,hidden,1,acFunct,0.90,0.98,XOR_INPUT,XOR_IDEAL,true);
		feedForward1.setEpoch(5000);

		feedForward1.iterate();
		feedForward1.printBestResult();

		
		
	}

}
