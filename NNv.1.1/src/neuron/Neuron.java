package neuron;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.Random;
import java.util.Vector;
import utils.BoundNumbers;
import activationFunction.ActivationFunction;


public class Neuron implements Serializable
{
	/**
	 * The neuron ID. 
	 */
	public int id;
	
	/**
	 *  The defined layer for the neuron. 
	 * 
	 *  for static structure a defined location 1=input layer, 2=hidden, 3=output
	 */
	public int layer;
		
	/**
	 * Connection reference for input neurons. 
	 */
	public Connection c;

	public double activationOutput;
	
	public double activationOutputSum;

	static int counter=0;
	
	public double error;
	
	public double input;

	public Connection[] cList2;
	
	public Map[] outputMap;
	
	public double delta;

	public Neuron()
	{
		this.id=counter();
	}
	public Neuron(int layer)
	{
		this.layer=layer;
		this.id=counter();
	}
	
	public double getInput() {
		return input;
	}
	public void setInput(double input) {
		this.input = input;
	}
	
	public Connection getC() {
		return c;
	}
	public void setC(Connection c) {
		this.c = c;
	}
	public Connection[] getCList2() {
		return cList2;
	}
	public void setCList2(Connection[] list2) {
		cList2 = list2;
	}
	public double getActivationOutput() {
		return activationOutput;
	}
	public void setActivationOutput(double activationOutput) {
		this.activationOutput = BoundNumbers.bound(activationOutput);
		//this.activationOutputSum+=this.activationOutput;
	}

	public int getId() {
		return id;
	}


	public void setId(int id) {
		this.id = id;
	}


	public int getLayer() {
		return layer;
	}


	public void setLayer(int layer) {
		this.layer = layer;
	}

	public static int getCounter() {
		return counter;
	}


	public static void setCounter(int counter) {
		Neuron.counter = counter;
	}
	
	public void setError(double error) {
		this.error = error;
	}
	
	public double getError() {
		return error;
	}
	
	public Map[] getOutputMap() {
		return outputMap;
	}
	public void setOutputMap(Map[] outputMap) {
		this.outputMap = outputMap;
	}
	
	public double getActivationOutputSum() {
		return activationOutputSum;
	}
	public void setActivationOutputSum(double activationOutputSum) {
		this.activationOutputSum = activationOutputSum;
	}
	
	/**
	 * A more sophisticated form of a feedforward method. 
	 * @param af
	 * @param neurons
	 */
	public void processInput(ActivationFunction af, boolean output){
		
		if(this.layer==1){
			for(Connection c:this.cList2){
				this.activationOutput=c.getInput();
			}
		}
		else
		{
			this.input=0.0;
			for(Connection c:this.cList2){
				this.input+=c.calculateWeightedInput();
			}
			if(!output){
				this.activationOutput=af.activationFunction(this.input);
			}
			else{
				this.activationOutput=this.input;

			}	
		}
		
		
	}

	public int counter()
	{
		return counter++;
	}
	
	
	@Override
	public String toString() {
		
		String output="Neuron  Id:"+this.id+" Layer: "+this.layer;
		
		output+=" Input :"+this.input;
		
		try
		{
		
		output+="    Connections:  ";
		
		if(this.layer != 1)
		{
			for(int i=0;i<this.cList2.length;i++)
			{
					output+=""+cList2[i].toString()+" ";
			}
		}
		
		output+="\n the output "+this.activationOutput;
		/*
		for(int l=0;l<this.outputMap.length;l++)
		{
			output+="\n outputs to: "+this.outputMap[l];
		}*/
		
		//output+=" Error: "+this.error;
		
		}catch (NullPointerException np)
		{
			System.out.println(np);
		}
	
		return output;
		
	}

	
}
