package FeedForwardNeuralNetwork;

import java.util.ArrayList;
import java.util.List;
import weka.core.Instance;

/**
 *
 * @author user-ari
 */
public class Neuron implements java.io.Serializable {
    
    public final static double DEFAULT_BIAS_VALUE = 1.0;
    
    // weights to apply to inputs (num inputs + 1 for the bias)
    protected List<Double> inputWeights;

    // derivatives of error in regard to weights
    protected List<Double> error;

    // last change in each weight
    protected List<Double> updatedWeights;
    
    //Real value (pecahan)
    protected double value;
    
    //Output Value (0,1)
    protected double output_value;
    
    
    // index in the weight vector of the bias weight (always at the end of the array)
    protected int biasIndex;
    
    public Neuron(){
        inputWeights = new ArrayList<Double>();
        error = new ArrayList<Double>();
        updatedWeights = new ArrayList<Double>();
        value = 0;
        
    }
    public double getOutputValue(){
        return output_value;
    }
    
    public void setOutputValue(double oval){
        output_value = oval;
    }
    
    public double getValue(){
        return value;
    }
    
    public void setValue(double val){
        value = val;
    }
    
    public void setError(double[] inputs){
        inputWeights.clear();
        for (int i=0; i < inputs.length ; i++){
            inputWeights.add(inputs[i]);
        }
    }
    
    public void setWeights(List<Double> inweights){
        //include bias weight
        inputWeights.clear();
        for (int i=0; i < inweights.size() ; i++){
            inputWeights.add(inweights.get(i));
        }  
    }
        
     public void setUpdatedWeights(List<Double> inweights){
        //include bias weight
        updatedWeights.clear();
        for (int i=0; i < inweights.size() ; i++){
            updatedWeights.add(inweights.get(i));
        }  
    }
    
    public double preactivation(Instance instance){
        double result = 0.0;
        double [] input = instance.toDoubleArray();
        int offset = 0;

        for(int i=0; i<input.length; i++)
        {
            // class values are not included
            if(i != instance.classIndex())
            {
                // never add missing values into the activation
                if(instance.isMissing(i))
                {
                    offset++;
                }
                else
                {
                    result += (input[i] * inputWeights.get(offset++));
                }
            }
        }

        // add the bias output
        result += (DEFAULT_BIAS_VALUE * inputWeights.get(inputWeights.size()-1).doubleValue());

        return result;
    }
    
    public double preactivation(double [] inputs)
    {
        // calculate the activation given an input vector

        double result = 0.0;

        for(int i=0; i<inputs.length; i++)
        {
            result += (inputs[i] * inputWeights.get(i).doubleValue());
        }

        // add the bias output
       // result += (DEFAULT_BIAS_VALUE * inputWeights.get(biasIndex).doubleValue());

        return result;
    }
    
    public double sigmoid(double x){
        return 1 / (1 + Math.exp(-x));
    }
    
    
    
    public double activate (Instance instance){
        return sigmoid(preactivation(instance));
    }
    
    public double activate (double[] inputs){
        return sigmoid(preactivation(inputs));
    }
    
    
    public List<Double> getError(){
        return error;
    }
    
    public List<Double> getLastWeightDeltas()
    {
        return updatedWeights;
    }

    public List<Double> getWeights()
    {
        return inputWeights;
    }

    public int getBiasIndex()
    {
        return biasIndex;
    }
    
 

}
