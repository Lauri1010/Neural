package neuron;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Vector;


/**
 * @author Lauri Turunen
 *
 * This class is used to manage the connections between the neurons
 * <br/><br/>
 * NOTE: This particular method ensures that the neural network is based on id values
 * and not a calculation of id based on a layer.
 */

public class NeuralMapping
{
	
	public NeuralMapping()
	{
		
	}
	
	/**
	 * Creates a mapping of the structure of the neural network
	 * 
	 * @param Neuron n
	 * @param inputs
	 * @param hidden
	 * @param outputs
	 * @return a Hashtable of inputs to neuron
	 */
	public Vector<Neuron> feedForwardSetNeuralNet(Vector<Neuron> n1,
			int nrInputs ,int[] nrHidden, int nrOutputs)
	{
		
		int a=-1;
		
		for(Neuron n:n1)
		{
	
			int id=n.getId();
			int layer=n.getLayer();
	
			if(layer==1)
			{
				Connection c=new Connection(layer);
				c.setInput(0.0);
				c.setWeight(1.0);
				Connection[] list2=new Connection[1];
				list2[0]=c;
				n.setCList2(list2);
				n.setC(c);
			}
			// Gets input from all input neurons. 
			if(layer==2)
			{
				int i=0;
				Connection[] list2=new Connection[nrInputs];
				int outputIndex=nrHidden[0]+nrInputs; // index to start charting outputs
				int outputLimit=nrHidden[0]+nrInputs+nrOutputs; // last index 
				int l=0;
				
				while(i < nrInputs)
				{
					Connection c=new Connection(layer);
					c.setThisNeuron(id);
					c.setFrom(i);
					//c.setOutputsTo(outputIndex);
					outputIndex++;	
	
					list2[i]=c;
					i++;
				}
				
				n.setCList2(list2);
				
			}
			if(layer==3)
			{
					Connection[] list3=new Connection[nrHidden[0]];
					int i= nrInputs;  // index of hidden layer
					int k=0; // index used for the array.
					int index=nrInputs+nrHidden[0];
	
					// only one hidden layer
					while(i<index)
					{
						Connection c=new Connection(layer);
						c.setThisNeuron(n.getId());
						c.setFrom(i);
						list3[k]=c;
						k++;
						i++;
					}
					n.setCList2(list3);
			}
			
				a++;
				n1.set(a, n);
				
		}

		return n1;

	}
	/*
	public Vector<Neuron> mapOutputs(Vector<Neuron> neurons, int lowerStart,int cutoff
			,int upperStart, int nrOfOutput)
	{
		int k=nrOfOutput;
		Map[] map;
		int i=lowerStart;

		do
		{
			Neuron n1=neurons.get(upperStart); // Next level neuron
			Neuron n2=neurons.get(i); // Hidden layer neuron
			map=new Map[k];
			Map temp=new Map();
			
			for(Connection c:n1.getCList2())
			{
				int j=0;
				if(c.getFrom()==i)
				{
					temp.setOutputsTo(n1.getId());
					map[j]=temp;
					j++;
				}
			}
			n2.setOutputMap(map);
			neurons.set(i, n2);
			i++;
		}
		while(i<cutoff);
		
		return neurons;
	}*/
	/**
	 * Creates the output mapping to each connection object for the neural network. 
	 */
	public Vector<Neuron> mapOututs(Vector<Neuron> neurons,int inputs,int hidden[],int outputs){
	
		int totalHidden=0;
		boolean l=false;
		int layers=0;
		
		for(int i=0;i<hidden.length;i++){
			totalHidden+=hidden[i];
			layers++;
		}
		/*
		if(totalHidden>hidden[0]){
			l=true;
		}*/
		
		for(Neuron n:neurons){
			if(n.layer<2){
					int index=hidden[0]+inputs-1;
					Map[] map=new Map[hidden[0]];
					for(int i=0;i<map.length;i++){
						Map m=new Map();
						
						for(int d=0;d<neurons.get(index).cList2.length;d++){
							Connection c=neurons.get(index).cList2[d];
							if(n.id==c.from){
								m.oc=d;
							}
						}
						
						m.outputsTo=index;
						index--;
						map[i]=m;
					}
					n.setOutputMap(map);
				}
			/*
			if(l){

			}*/
			
			if(n.layer>1 && n.layer<3){
				Map[] map=new Map[outputs];
				int index=inputs+hidden[0]+outputs-1;
				for(int i=0;i<map.length;i++){
					Map m=new Map();
					
					for(int d=0;d<neurons.get(index).cList2.length;d++){
						Connection c=neurons.get(index).cList2[d];
						if(n.id==c.from){
							m.oc=d;
						}
					}
					
					m.outputsTo=index;
					index--;
					map[i]=m;
				}
				n.setOutputMap(map);
			}
		}
		
		return neurons;
		
	}

	
	
	

}
