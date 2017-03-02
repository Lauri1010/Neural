package simulatedAnnealing;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Vector;

import activationFunction.ActivationFunction;
import neuron.Connection;
import neuron.Neuron;
import supervisedLearning.BackPropagationNetwork;
import utils.ObjectCloner;
import utils.SerializeObject;

public class LearningSystem extends NeuralSimulatedAnnealing {
	
	private double startTemp;
	private double stoptemp;
	private int cycles;
	private BackPropagationNetwork network;
	private Vector<Neuron> bestSet;
	private double currentTemp;
	private double error;
	// For each a new intialization of weights... 
	
	public LearningSystem(int inputs, int hidden[], int output,ActivationFunction acFunction,
			double learningRate, double momentum, final double[][] input, final double[][] ideal,boolean train,
			double startTemp, double stoptemp, int cycles){
		
		super();
		this.startTemp = startTemp;
		this.stoptemp = stoptemp;
		this.cycles = cycles;
		
	}
	
	@SuppressWarnings("unchecked")
	public void setNetworkAndCopyBestSet(BackPropagationNetwork network) throws Exception{
		
		this.network = network;
		this.bestSet = (Vector<Neuron>) ObjectCloner.deepCopy(network.getNeurons());
		
	}
	
	

	/**
	 * @throws Exception
	 * Implements the Simulated annealing algorithm. For this method to be used a network represented in the form of a vector
	 * needs to be set to this class. 
	 */
	@SuppressWarnings("unchecked")
	public void iterate() throws Exception{
		this.cycles++;
		double bestError=this.network.getBestErrorRate();
		this.bestSet = (Vector<Neuron>) ObjectCloner.deepCopy(network.getNeurons());
		// Error here..
		currentTemp=startTemp;
		boolean found=false;
		int c=0;
		double error=0.0;
		do{
			do{
				if(c==0)this.network.setNeurons((Vector<Neuron>) ObjectCloner.deepCopy(bestSet));
				randomize(this.network.getNeurons());
				this.network.run(false);
			
				error=this.network.getTotalError();
				setError(error);
	
				if(error<bestError){
					bestError=error;
					this.bestSet=(Vector<Neuron>) ObjectCloner.deepCopy(this.network.getNeurons());
					found=true;
					printResults();
				}
				c++;

				this.network.setTotalError(0.0);
			}while(c<cycles);
			c=0;
		final double ratio = Math.exp(Math.log(this.stoptemp / this.startTemp) / (this.cycles - 1));
		this.currentTemp *= ratio;
			
		}while(currentTemp>stoptemp);
		
		intitalizeWeights((Vector<Neuron>) ObjectCloner.deepCopy(bestSet));
		this.network.setNeurons((Vector<Neuron>) ObjectCloner.deepCopy(bestSet));
		
	}
	
	/**
	 * @throws Exception
	 * Implements the Simulated annealing algorithm. For this method to be used a network represented in the form of a vector
	 * needs to be set to this class. 
	 */
	@SuppressWarnings("unchecked")
	public BackPropagationNetwork iterateAndReturnNetwork() throws Exception{
		this.cycles++;
		double bestError=this.network.getBestErrorRate();
		this.bestSet = (Vector<Neuron>) ObjectCloner.deepCopy(network.getNeurons());
		// Error here..
		currentTemp=startTemp;
		boolean found=false;
		int c=0;
		double error=0.0;
		do{
			do{
				if(c==0)this.network.setNeurons((Vector<Neuron>) ObjectCloner.deepCopy(bestSet));
				randomize(this.network.getNeurons());
				this.network.run(false);
			
				error=this.network.getTotalError();
				setError(error);
	
				if(error<bestError){
					bestError=error;
					this.bestSet=(Vector<Neuron>) ObjectCloner.deepCopy(this.network.getNeurons());
					found=true;
					printResults();
				}
				c++;

				this.network.setTotalError(0.0);
			}while(c<cycles);
			c=0;
		final double ratio = Math.exp(Math.log(this.stoptemp / this.startTemp) / (this.cycles - 1));
		this.currentTemp *= ratio;
			
		}while(currentTemp>stoptemp);
		
		intitalizeWeights((Vector<Neuron>) ObjectCloner.deepCopy(bestSet));
		this.network.setNeurons((Vector<Neuron>) ObjectCloner.deepCopy(bestSet));
		
		return this.network;
		
	}
	
	/**
	 * @param neurons
	 * 		The neural network neurons
	 * 
	 * Used to randomize the weights of the network. 
	 * 
	 */
	public void randomize(Vector<Neuron> neurons){
		for(Neuron n:neurons){
			if(n.getLayer()>1){
				for(Connection c:n.getCList2()){
					double add = 0.5 - (Math.random());
					add /= this.startTemp;
					add *= this.currentTemp;
					c.setWeight(c.getWeight()+add);
					c.setpWeight(c.getWeight()+add);
				}
			}
		}
	}
	
	public void printResults(){
		System.out.println("Annealing best total error: "+this.network.getTotalError());
	}

	public double getStartTemp() {
		return startTemp;
	}

	public void setStartTemp(int startTemp) {
		this.startTemp = startTemp;
	}

	public double getStoptemp() {
		return stoptemp;
	}

	public void setStoptemp(int stoptemp) {
		this.stoptemp = stoptemp;
	}

	public int getCycles() {
		return cycles;
	}

	public void setCycles(int cycles) {
		this.cycles = cycles;
	}

	public BackPropagationNetwork getNetwork() {
		return network;
	}
	
	public Vector<Neuron> getBestSet() {
		return bestSet;
	}

	public void setBestSet(Vector<Neuron> bestSet) {
		this.bestSet = bestSet;
	}

	public void setNetwork(BackPropagationNetwork network) {
		this.network = network;
	}
	/**
	 * A method for deep cloning objects. 
	 * 
	 * @param n  The neural network
	 * @return A cloned neural network
	 */
	  public Vector<Neuron> clone(Vector<Neuron> n) {
		    Object clonedObj = null;
		    try {
		      ByteArrayOutputStream baos = new ByteArrayOutputStream();
		      ObjectOutputStream oos = new ObjectOutputStream(baos);
		      oos.writeObject(n);
		      oos.close();

		      ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
		      ObjectInputStream ois = new ObjectInputStream(bais);
		      clonedObj = ois.readObject();
		      ois.close();
		    } catch (Exception cnfe) {
		      System.out.println("Class not found " + cnfe);
		    }
		    return (Vector<Neuron>) clonedObj;

		  }

	public void setError(double error) {
		this.error = error;
	}

	public double getError() {
		return error;
	}

	public double getCurrentTemp() {
		return currentTemp;
	}

	public void setCurrentTemp(double currentTemp) {
		this.currentTemp = currentTemp;
	}

	public void setStartTemp(double startTemp) {
		this.startTemp = startTemp;
	}

	public void setStoptemp(double stoptemp) {
		this.stoptemp = stoptemp;
	}
	/**
	 * Initializes the weights for the purpose of restarting the system
	 */
	public void intitalizeWeights(Vector<Neuron> neurons){	
		for(Neuron n:neurons){
			if(n.getLayer()>1){
				for(Connection c:n.getCList2()){
					c.intitalizePreviousWeights();
				}
			}
		}
		
		
	}

}
