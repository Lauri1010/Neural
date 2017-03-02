package supervisedLearning;

import java.text.NumberFormat;
import java.util.Hashtable;
import java.util.Vector;
import neuron.Connection;
import neuron.Map;
import neuron.NeuralMapping;
import neuron.Neuron;
import activationFunction.ActivationFunction;
import utils.BoundNumbers;
import utils.ObjectCloner;

/**
 * @author Lauri Turunen 
 * <br/><br/>
 *
 */

public class BackPropagationNetwork
{

	private int j=0; 
	/**
	 * Used to call neural mapping class methods.  
	 */
	private NeuralMapping nm;
	/**
	 * The number of input neurons in the network.
	 */
	private int thisInputs=0;
	/**
	 * The number of hidden neurons in the network (an array because there may be more than one).
	 */
	private int thisHidden[]=new int[2];
	/**
	 * The number of output neurons in the neural network. 
	 */
	private int thisOutputs=0;
	/**
	 * The data location used to reference for input and ideal data. 
	 */
	public int dataLocation=0;
	/**
	 * The 
	 * neuron vector used to store the different neurons
	 */
	protected Vector<Neuron> neurons;
	/**
	 * The total error (RMS) for the entire training set. 
	 */
	private double totalError=0.0;
	/**
	 * The best error rate thus far 5000000.0 for comparison purposes. 
	 */
	private double bestErrorRate=5000000.0;
	/**
	 * A best set vector that can be used to store a best result of the training. 
	 */
	Vector<Neuron> bestSet; 
	/**
	 * the starting point for layer 1 neurons.
	 */
	int i1;
	/**
	 * The cutoff point for layer 1 neurons. 
	 */
	int t1;
	/**
	 * The starting index for layer 2 neurons
	 */
	int i2;
	/**
	 * The cutoff point for layer 2 neurons. 
	 */
	int t2;
	/**
	 * The starting point for layer 3 neurons
	 */
	int i3;
	/**
	 * The cutoff point for layer 3 neurons. 
	 */
	int t3;
	/**
	 * The number of epochs (used with iterate method). 
	 */
	private int epoch;
	/**
	 * A variable representing the current iteration.
	 */
	int p=0;
	/**
	 * Stores the weight sums for the upper layer. Used for backpropagation.
	 * References the lower level weight to the weight of the upper layer neuron connection. 
	 */
	private Hashtable<Integer, Double> weightSum;
	/**
	 * Stores the upper neuron connection  delta value. 
	 */
	private Hashtable<Integer, Double> upperDelta;
	/**
	 * Stores the error data for output layer neuron(s).
	 */
	private Hashtable<Integer, Double> errorData;
	/**
	 * Holds the data of the actual prediction data (when predicting with the real test data). 
	 */
	private double[] actualDataReturn;
	/**
	 * The current error for the iteration (actual - desired). 
	 */
	private double currentError;
	/**
	 * The activation function variable (interface class instance). 
	 */
	protected ActivationFunction acFunct;
	/**
	 * The desired result data (in training). 
	 */
	protected double[][] ideal;
	/**
	 * The given input data. 
	 */
	protected double[][] input;
	/**
	 * The actual data input for the testing phase. 
	 */
	protected double[] actualDataInput;
	/**
	 * The learning rate (alpha)
	 */
	protected double lRate;
	/**
	 * The momentum; how much past weights affect the weigh change calculation).
	 */
	protected double momentum;
	/**
	 * The output layer size (used in predicting with actual data). 
	 */
	protected int outputSize;
	
	NumberFormat f;
	
	double traningValueSum;
	
	double bPercentage;
	
	/**
	 * The error percentage
	 */
	double peError;
	
	/**
	 * The default momentum
	 */
	double dMomentum;
	
	/**
	 * The default learning rate
	 */
	double dLearningRate;
	
	double lRdescent;
	
	double mDescent;
	
	double firstValue;
	
	boolean fError;
	
	/**
	 * Creates a Backpropagation network using the parameters.
	 * 
	 * @param inputs  the number of input neurons
	 * @param hidden the number of hidden neurons
	 * @param output the number of output neurons
	 * @param acFunction the activation function. 
	 * @param learningRate the learning rate
	 * @param momentum the momentum (how much past weight affects the training)
	 * @param input the input data
	 * @param ideal the ideal data
	 * @param train determines whether the system is on full trainig mode
	 */
	public BackPropagationNetwork(int inputs, int hidden[], int output,ActivationFunction acFunction,
			double learningRate, double momentum, final double[][] input, final double[][] ideal,boolean train)
	{
		thisInputs=inputs;
		thisHidden=hidden;
		thisOutputs=output;
		this.acFunct=acFunction;
		this.momentum=momentum;
		this.input=input;
		this.ideal=ideal;
		this.lRate=learningRate;
		weightSum=new Hashtable<Integer,Double>();
		upperDelta=new Hashtable<Integer, Double>();
		errorData=new Hashtable<Integer, Double>(20,20);
		neurons = new Vector<Neuron>();
		bestSet=new Vector<Neuron>();
		epoch=0;
		this.f= NumberFormat.getPercentInstance();  
		this.f.setMinimumFractionDigits(1);  
		this.f.setMaximumFractionDigits(1); 
		this.dMomentum=momentum;
		this.dLearningRate=learningRate;
		this.lRdescent=5.0;
		this.mDescent=1.0;
		this.fError=true;
		
		i1=0;
		t1=thisInputs;
		i2=thisInputs;
		t2=thisHidden[0]+thisInputs;
		i3=thisHidden[0]+thisInputs;
		t3=thisHidden[0]+thisInputs+thisOutputs;
		
		for(int i=0;i<inputs;i++)
		{
			neurons.add(new Neuron(1));
		}
		for(int j=0;j<hidden[0];j++)
		{
			neurons.add(new Neuron(2));
		}
		for(int k=0;k<output;k++)
		{
			neurons.add(new Neuron(3));
		}
		
		this.nm=new NeuralMapping();
		createNetwork();
		if(train){calclulateTraningValueSum();}
	}
	/**
	 * Creates a neural network using the data of number of neurons in each layer. 
	 */
	public void createNetwork(){
		neurons=nm.feedForwardSetNeuralNet(neurons,thisInputs,thisHidden,thisOutputs);		
		neurons=nm.mapOututs(neurons, thisInputs, thisHidden, thisOutputs);
		System.out.println("\n Network created \n");
	}
	/**
	 * Iterates the network. there are three phases: <br/><br/>
	 * Phase A: <br/>
	 * 1. input patterns<br/>
	 * 2. feedforward: Pass the values forward (with defined weights)<br/>
	 * 3. backpropagate: propagate backwards and calclulate the error sum for each connection.<br/>
	 * 4. Manage results: increment total error <br/><br/>
	 * Phase B: <br/>
	 * 1. Learn: adjust weights with the error sum <br/>
	 * 2. Store best result and print the current total error. 
	 * @throws Exception 
	 */
	public void iterate() throws Exception
	{
		do
		{
		  this.p++;
			
			do
			{
				inputPatterns();
				feedForward(false);
				backPropagate();
				manageResults();
				dataLocation++;
			}
			while(dataLocation<input.length);
			
			dataLocation=0;
			Learn();
			calculateErrorPercentage();
			printResult();
			testBestSet();
			errorData.clear();
			this.peError=0.0;
		}
		while(this.p<epoch);

	}
	/**
	 * Does one iteration of the network. there are three phases: <br/><br/>
	 * Phase A: <br/>
	 * 1. input patterns<br/>
	 * 2. feedforward: Pass the values forward (with defined weights)<br/>
	 * 3. backpropagate: propagate backwards and calclulate the error sum for each connection.<br/>
	 * 4. Manage results: increment total error <br/><br/>
	 * Phase B: <br/>
	 * 1. Learn: adjust weights with the error sum <br/>
	 * 2. Store best result and return the current error. 
	 * @throws Exception 
	 */
	public double iteration() throws Exception
	{
		do
		{
			inputPatterns();
			feedForward(false);
			backPropagate();
			manageResults();
			dataLocation++;
		}
		while(dataLocation<input.length);
		dataLocation=0;
		calculateErrorPercentage();
		double error=this.peError;
		testBestSet();
		Learn();
		errorData.clear();
		this.peError=0.0;
		return error;
	}
	
	/**
	 * A method that can be used only for the feedfoward. Useful for other methods than backpropagation. 
	 * 
	 * @param details
	 * 		whether to show the errors with each training case
	 */
	
	public void run(boolean details)
	{
	
		do
		{
			inputPatterns();
			feedForward(false);
			manageResults();
			if(details){System.out.println("" +
			"the desired value "+ideal[this.dataLocation][0]+"  The current output "
			+Math.sqrt(this.neurons.lastElement().getActivationOutput())+" ");}
			this.dataLocation++;
		}
		while(this.dataLocation<input.length);
		
		this.dataLocation=0;
	}
	
	/**
	 * Used to input patterns for the input layer neurons
	 */
	public void inputPatterns() {
		
		int a=0;
		int i=-1;
		
		for(Neuron n: neurons)
		{
			i++;
			if(n.getLayer()==1)
			{
				n.getC().setInput(BoundNumbers.bound(input[dataLocation][a]));
				a++;
				if(a%thisInputs==0)
				{	
					a=0;
				}
				neurons.set(i, n);
			}

		}
	}

	/**
	 * Backpropagates thrugh the network and calculates the adjustments of weights.
	 * <br/><br/>
	 * For each output layer neuron connection a call is made to setOutputError(c,n).
	 * For each Hidden layer neuron a call is made to setHiddenError(c,n). 
	 * 
	 */
	public void backPropagate() {
		
		for(int i=thisInputs+thisHidden[0]+thisOutputs-1;
		i>thisInputs+thisHidden[0]-1;i--){
			Neuron n=neurons.get(i);
			if(n.getLayer()==3){
				for(Connection c:n.getCList2()){
					setOutputError(c,n);
				}
			}
		}
		for(int i=thisInputs;i<thisInputs+thisHidden[0];i++){
			Neuron n=neurons.get(i);
			if(n.getLayer()==2){
				for(Connection c:n.getCList2()){
					setHiddenError(c,n);
				}
			}
			
		}
	}
	/**
	 * @param n
	 * 	The output neuron used to calculate the error.
	 *  Note not customized for more than one output neuron.  
	 */
	public void setError(Neuron n) {
		double error=n.activationOutput-ideal[this.dataLocation][0];  
		this.currentError=Math.pow(error, 2);
		errorData.put(n.id,error);
	}

	
	/**
	 * Sets the output layer error and stores a reference to upperDelta Hashtable for further calculations
	 * Sums the error data for the output layer connections.<br/><br/>
	 * The calclulation: <br/>
	 * 
	 * Delta: error * Activation of summed input <br/><br/>
	 * Error sum: delta * activation output of the previous layer neuron (on feedforward). <br/>
	 * 
	 * <br/>
	 * @param c the current connection
	 * @param n the current neuron object
	 */
	public void setOutputError(Connection c, Neuron n) { 
		double delta=errorData.get(c.thisNeuron)*acFunct.derivativeFunction(n.input);
		c.sumError(delta*neurons.get(c.from).getActivationOutput());
		upperDelta.put(c.from, delta);
	}
	/**
	 * Sets the hidden layer Delta value combined with the activation output multiplication of the upper layer
	 * <br/><br/>
	 * The calculation:
	 * Delta: sum of all weights to conenction between hidden and output neurons * derivate of summed input to neuron<br/><br/>
	 * weight sum: previous layer neurons activation output (in feedforward) * (-delta)
	 * <br/>
	 * @param c  the current connection
	 * @param n  the current neuron object
	 */
	public void setHiddenError(Connection c, Neuron n){		 
		double delta=upperDelta.get(c.thisNeuron)*weightSum.get(c.thisNeuron)*acFunct.derivativeFunction(n.input);
		c.sumError((neurons.get(c.from).getActivationOutput())*-delta); 
	}
	/**
	 * A method used to adjust the weights of the neural network based on the stored data. After each call the error for each neuron 
	 * is zeroed out.
	 */
	public void Learn()
	{
		for(Neuron n:neurons)
		{
			if(n.getLayer()!=1)
			{
				for(Connection c:n.getCList2())
				{
					c.adjustWeights(lRate,momentum);
					c.clearError();
				}
			}
		}
	}
	
	public void outputNeuronsToConsole()
	{ 
		for(Neuron n:neurons)
		{
			System.out.println(n.toString());
			System.out.println("");
		}
	}
	/**
	 * A method for testing and saving the best error results.
	 */
	@SuppressWarnings("unchecked")
	public void testBestSet() throws Exception{
		
		if(totalError<bestErrorRate)
		{
			bestErrorRate=totalError;
			this.bPercentage=this.peError;
			this.peError=0.0;
			totalError=0.0;
			currentError=0.0;
		}
		else
		{
			totalError=0.0;
			currentError=0.0;
			this.peError=0.0;
		}
	}
	
	/**
	 * A feedforward method using output maps. Passes the input values forward based on the neuron mappings <br/><br/>
	 * 	Note this method requires the neurons to be in correct order in the vector
	 * @param actual
	 */
	public void feedForward(boolean actual){
		
		for(Neuron n:neurons){
			if(n.getLayer()<2+thisHidden.length){
				n.processInput(this.acFunct, false);
				for(Map m:n.outputMap){
					Connection c=neurons.get(m.outputsTo).cList2[m.oc];
					c.input=n.activationOutput;
				}
			}
			else{
				n.processInput(this.acFunct, true);
				for(Connection c:n.getCList2()){
					weightSum.put(c.from, c.weight);
				}
				if(!actual){setError(n);}
			}
		}
	}
	/**
	 * Used to feed actual prediction data to the system
	 * 
	 * @return
	 */
	public double[] actualDataPrediction()
	{
		int e=0;
		
		do{
			if(dataLocation<this.actualDataInput.length){
				inputSinglePattern(this.actualDataInput);
				feedForward(true);
				actualDataReturn[e]=neurons.lastElement().getActivationOutput();
			}
			e++;
		}
		while(e<this.outputSize);
		
		return actualDataReturn;
	}
	
	public void inputSinglePattern(double[] pattern){
		dataLocation=0;
		for(Neuron n: neurons){
			if(n.getLayer()==1){
				n.getC().setInput(pattern[dataLocation]);
				dataLocation++;
			}
		}
	}
	
	public void manageResults()
	{
		this.totalError+=Math.sqrt(currentError);
	}
	public void printResult(){
		System.out.println("Iteration: "+this.p+" Error:  "+this.peError+" ");  
	}
	/**
	 * calclulates the error percentage based on the first value (in decimal format). 
	 */
	public void calculateErrorPercentage(){ 
		if(fError){
			this.firstValue=this.totalError;
			fError=false;
			this.peError=1.0;
		}else{
			this.peError=this.totalError/this.firstValue;
		}
	
		
	}
	public void calclulateTraningValueSum(){
		for(int i=0;i<this.ideal.length;i++){
			this.traningValueSum+=this.ideal[i][0];
		}
	}
	
	public void setEpoch(int epoch)
	{
		this.epoch=epoch;
	}
	
	public void printBestResult() throws Exception{
		if(this.bPercentage<0.001){
			System.out.println("Best error result: 0.0 % ");
		}else{
			System.out.println("Best error result: "+f.format(this.bPercentage)+" ");
		}
	}

	public double getTotalError() {
		return totalError;
	}

	public void setTotalError(double totalError) {
		this.totalError = totalError;
	}

	public int getInputLocation() {
		return dataLocation;
	}

	public void setInputLocation(int inputLocation) {
		this.dataLocation = inputLocation;
	}

	public double getBestErrorRate() {
		return bestErrorRate;
	}

	public void setBestErrorRate(double bestErrorRate) {
		this.bestErrorRate = bestErrorRate;
	}
	
	@SuppressWarnings("unchecked")
	public void setNeurons(Vector<Neuron> neurons) throws Exception {
		this.neurons = (Vector<Neuron>) ObjectCloner.deepCopy(neurons);
	}

	public Vector<Neuron> getNeurons() {
		return neurons;
	}

	public double[] getActualDataReturn() {
		return actualDataReturn;
	}

	public void setActualDataReturn(double[] actualDataReturn) {
		this.actualDataReturn = actualDataReturn;
	}
	public double[] getActualDataInput() {
		return actualDataInput;
	}
	public void setActualDataInput(double[] actualDataInput) {
		this.actualDataInput = actualDataInput;
	}
	public int getOutputSize() {
		return outputSize;
	}
	public void setOutputSize(int outputSize) {
		this.outputSize = outputSize;
	}
	public double getPeError() {
		return peError;
	}
	public String getErrorPercentage(){
		return f.format(this.peError);
	}
	
	public void setPeError(double peError) {
		this.peError = peError;
	}

	public double getMomentum() {
		return momentum;
	}
	public void setMomentum(double momentum) {
		this.momentum = momentum;
	}
	public double getLRate() {
		return lRate;
	}
	public void setLRate(double rate) {
		lRate = rate;
	}
	public void decreaseLearningRate(){
		if(this.lRate>0.000001){
		this.lRate*=0.2;}
	}
	public void increaseLearningRate(double rate){
		if(this.lRate<0.0015){
		this.lRate*=rate;}
	}
	public void largeMomentumIncrease(){
		this.momentum+=0.001;
	}
	public void increaseSmallLearningRate(){
		this.lRate+=0.0000004;
	}
	
	public void increaseLearningRate2(double max){
		if(this.lRate<max){this.lRate*=1.5;}
	}
	
	public void increaseLearningRate3(double max){
		if(this.lRate<max){this.lRate*=1.08;}
	}
	public void increaseMomentum(double maximum){
		if(this.momentum<maximum){
		this.momentum*=2.0;}
	}
	public void decreaseMomentum(){
		if(this.momentum>0.0000001){
		this.momentum*=0.05;}
	}
	public void setDefaultMomentum(){
		this.momentum=dMomentum;
	}
	
	public void setLearningRate(double lRate){
		
		this.lRate=lRate;
		
	}
	
	public void setDefaultLearningRate(){
		this.lRate=this.dLearningRate+0.0001;
	}
	public void setBestNeurons() throws Exception{
		this.neurons=(Vector<Neuron>) ObjectCloner.deepCopy(bestSet);
	}
	public void setRandomRates(){
		this.lRate=0.00001 * (double)Math.random() - 0.05;
		this.momentum=0.00001 * (double)Math.random() - 0.05;
	}
	public void manageLearningRate(){
		this.lRdescent*=0.99999995;
		this.lRate=this.dLearningRate*this.lRdescent;
	}
	
	
}
