package neuron;

import java.io.Serializable;

import utils.BoundNumbers;

public class Connection extends Map implements Serializable{

	/**
	 * The Serializable serial version id. 
	 */
	private static final long serialVersionUID = 8296844575310088102L;
	/**
	 * This neuron (based on id)
	 */
	public int thisNeuron;
	/**
	 * Input to connection
	 */
	public double input;
	/**
	 * Output from connection
	 */
	public double output;
	/**
	 * value from what neuron (id). 
	 */
	public int from;
	/**
	 * The current error. 
	 */
	public double error;
	public double sum;
	public double delta;
	public double wResult; 
	public double weight;
	/**
	 * The previous weight
	 */
	public double pWeight; 
	
	public Connection(int layer) {
		super();
		if(layer!=1)
		{
		this.weight = 0.1 * (double)Math.random() - 0.05;
		}
		else
		{
			this.weight= 1.0;
		}
		this.pWeight=this.weight;
	}
	
	public double getWResult() {
		return wResult;
	}

	public void setWResult(double result) {
		wResult = result;
	}
	
	public int getOutputsTo() {
		return outputsTo;
	}

	public void setOutputsTo(int outputsTo) {
		this.outputsTo = outputsTo;
	}
	
	public double getSum() {
		return sum;
	}

	public void setSum(double sum) {
		this.sum = BoundNumbers.bound(sum);
	}
	
	public double getInput() {
		return input;
	}

	public void setInput(double input) {
		this.input = BoundNumbers.bound(input);
	}

	public double getOutput() {
		return output;
	}

	public void setOutput(double output) {
		this.output = BoundNumbers.bound(output);
	}

	public double getError() {
		return error;
	}

	public void setError(double error) {
		this.error = BoundNumbers.bound(error);
	}
	
	public void sumError(double error){
		this.error+=BoundNumbers.bound(error);
	}

	public double getWeight() {
		return weight;
	}

	public void setWeight(double weight) {
		this.weight = BoundNumbers.bound(weight);
	}	
	public int getFrom() {
		return from;
	}

	public void setFrom(int from) {
		this.from = from;
	}
	
	public int getThisNeuron() {
		return thisNeuron;
	}

	public void setThisNeuron(int thisNeuron) {
		this.thisNeuron = thisNeuron;
	}
	
	public void setDelta(double delta) {
		this.delta = BoundNumbers.bound(delta);
	}

	public double getDelta() {
		return delta;
	}
	
	public double getpWeight() {
		return pWeight;
	}

	public void setpWeight(double pWeight) {
		this.pWeight = pWeight;
	}

	public double calculateWeightedInput()
	{
		this.wResult=this.input * this.weight;
		return wResult;
	}
	
	public void adjustWeights(double learningRate,double momentum)
	{
		// w-1 * (previous wight).
		// pWeight= previous weight
		
		double a=this.weight;
		
		this.weight +=((learningRate*this.error)+(momentum*(this.weight-this.pWeight)));
		
		// first round same as current weight;
		this.pWeight=a;
	}
	
	public void intitalizePreviousWeights(){
		this.pWeight = this.weight;
	}
	
	public void clearError()
	{
		this.error=0.0;
	}
	
	public int passOutput()
	{
		return 1;
	}

	@Override
	public String toString() {
		
		
		String output = "";
		
		output="\n input from:  "
		+this.from
		+" The input: "
		+this.input
		+" This Neuron: "
		+this.thisNeuron
		+" Weight: "
		+this.weight
		+" Error: "
		+this.error;

		return output;
		
	}


	
	
}
