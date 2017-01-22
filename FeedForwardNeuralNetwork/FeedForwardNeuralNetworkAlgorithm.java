package FeedForwardNeuralNetwork;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class FeedForwardNeuralNetworkAlgorithm implements java.io.Serializable {
    public final static double UPPER_THRESHOLD = +45.0;
    public final static double LOWER_THRESHOLD = -45.0;
    public final static double MAX = +1.0;
    public final static double MIN =  0.0;
    
    protected Instances origin;
    protected Instances instances;
    protected Neuron[][] neurons;
    protected double sumError;
    protected int hidden_layers;
    protected double learning_rate = 0.08;
    private final RandomWrapper rnd = new RandomWrapper();
    
//Class Output
    protected double class_output_value;
    

    public void buildModel(int n_hidden_layer, int neuron_hidden_layer){
        hidden_layers = n_hidden_layer;
        //Ini weight, belum sesuai input
        List<Double> arr = new ArrayList<>();
        
        
        //count total number of neurons
        int num_neurons = 0;
        switch (n_hidden_layer) {
            case 0:
                num_neurons = (instances.numAttributes()-1)+ instances.numClasses();
                break;
            case 1:
                num_neurons = (instances.numAttributes()-1)+ instances.numClasses() + neuron_hidden_layer;
                break;
            default:
                throw new RuntimeException("Illegal n_hidden_layer");
        }
        //Build array container
        neurons = new Neuron[n_hidden_layer+2][];
        //Input Layer
        neurons[0] = new Neuron[instances.numAttributes()-1];
        for (int j=0; j< instances.numAttributes()-1; j++){
                neurons[0][j] = new Neuron();
        }
        
        //Hidden Layer
        neurons[1] = new Neuron[neuron_hidden_layer];
        if (n_hidden_layer == 1){
            for (int j=0; j< neuron_hidden_layer; j++){
                neurons[1][j] = new Neuron();
                for (int i=0; i<neurons[0].length; i++){
                    double temp = rnd.getRand().nextDouble();
                    arr.add(temp);
                }
                neurons[1][j].setWeights(arr);
                arr.clear();
            }
            //Output Layer
            neurons[2] = new Neuron[instances.numClasses()];
            for (int j=0; j< instances.numClasses(); j++){
                neurons[2][j] = new Neuron();
                for (int i=0; i<neurons[1].length; i++){
                    double temp = rnd.getRand().nextDouble();
                    arr.add(temp);
                }
                neurons[2][j].setWeights(arr);
                arr.clear();
            }
        }
        else //Hidden layer = 0
        {
            neurons[1] = new Neuron[instances.numClasses()];
            for (int j=0; j< instances.numClasses(); j++){
                neurons[1][j] = new Neuron();
                for (int i=0; i<neurons[0].length; i++){
                    double temp = rnd.getRand().nextDouble();
                    arr.add(temp);
                }
                neurons[1][j].setWeights(arr);
                arr.clear();
            }
        }
        
        
        
    }
    
    public void clearModel(){
        for (int i=0 ; i<neurons.length; i++){
            for(int j=0; j<neurons[i].length; j++){
                neurons[i][j].setValue(0);
            }
        }
    }
    
    public void printModel(){
        for (int i = 0 ; i< neurons.length; i++){
            System.out.print("Layer "+i+" : ");
            for (int j =0 ; j < neurons[i].length ; j++){
                System.out.print(neurons[i][j].getValue()+" ");
                if (i==neurons.length-1)
                    System.out.println("FinalOut : "+neurons[neurons.length-1][j].getOutputValue());
            }
            System.out.println("");
        }
       
    }
    
    public FeedForwardNeuralNetworkAlgorithm(Instances i){
		instances = i;
                hidden_layers = -1;
                sumError = 1;
    }
    
    public void setInputLayer(double[] inputs ){
        //Asumsi sudah sama panjangnya
        for (int i=0 ; i < neurons[0].length ; i++){
            neurons[0][i].setValue(inputs[i]);
        }
    } 
    
    public void updateModel(Instance curr_instance){
        sumError = countThresholdError(curr_instance);
        if (hidden_layers == 0){
            double[] error = countOutputError(curr_instance);
            for (int i=0 ; i<neurons[1].length; i++){
                List<Double> current_weights = new ArrayList<>(neurons[1][i].getWeights());
                for (int j=0; j<current_weights.size() ; j++){
                    double new_weight = current_weights.get(j) + learning_rate * error[i] * neurons[0][j].getValue();
                    current_weights.set(j, new_weight);  
                }
                neurons[1][i].setWeights(current_weights);
            }
            
        }
        else if (hidden_layers == 1)
        {
            //Update input weights of output layer
            double[] errorOutput = countOutputError(curr_instance);
            for (int i=0 ; i<neurons[2].length; i++){
                List<Double> current_weights = new ArrayList<>(neurons[2][i].getWeights());
                for (int j=0; j<current_weights.size() ; j++){
                    double new_weight = current_weights.get(j).doubleValue() + learning_rate * errorOutput[i] * neurons[1][j].getValue();
                    current_weights.set(j, new Double(new_weight));  
                }
                neurons[2][i].setWeights(current_weights);
            }
            //Update input weights of hidden layer
            double[] errorHidden = countHiddenError(curr_instance);
            for (int i=0 ; i<neurons[1].length; i++){
                List<Double> current_weights = new ArrayList<>(neurons[1][i].getWeights());
                for (int j=0; j<current_weights.size() ; j++){
                    double new_weight = current_weights.get(j).doubleValue() + learning_rate * errorHidden[i] * neurons[0][j].getValue();
                    current_weights.set(j, new Double(new_weight));  
                }
                neurons[1][i].setWeights(current_weights);
            }
            
        }
    }
    
    public double countThresholdError(Instance instance){
        int classnum = neurons[neurons.length-1].length;
        double result = 0;
        for (int i=0; i<classnum; i++){
            double expectedValue;
            if (i == instance.classValue()) 
            {
                expectedValue = 1.0;
            }
            else
            {
                expectedValue = 0.0;
            }
            result += Math.pow(expectedValue-neurons[neurons.length-1][i].getValue(), 2) / 2;
        }
        return result;
        
    }
    
    
    public double[] countOutputError(Instance instance){
        //Only invoke this after call countOutput method
        int classnum = neurons[neurons.length-1].length;
        double[] result = new double[classnum];
        for (int i=0; i<classnum; i++){
            double expectedValue;
            if (i == instance.classValue()) 
            {
                expectedValue = 1.0;
            }
            else
            {
                expectedValue = 0.0;
            }
            result[i] = neurons[neurons.length-1][i].getValue() * (1-neurons[neurons.length-1][i].getValue()) * (expectedValue - neurons[neurons.length-1][i].getValue());
        }
        return result;
    }
    
    public double[] countHiddenError(Instance instance){
        //only call this if there's hidden layer
        int hidnum = neurons[1].length;
        double[] result = new double[hidnum];
        
        //count sum of error * weight first
        int classnum = neurons[neurons.length-1].length;
        double sum = 0;
        double[] error = countOutputError(instance);
        for (int k=0; k<hidnum; k++){
            for (int i=0; i<classnum; i++){
                sum += error[i]*neurons[2][i].getWeights().get(k);
            }
            double out = neurons[1][k].getValue();
            result[k] = out * (1-out) * (sum);
        }
        
        /*
        for (int i=0; i<classnum; i++){
            List<Double> weights = neurons[2][i].getWeights();
            for(int j=0; j<weights.size() ; j++){
                sum += error[i] * weights.get(j);
            }
            for (int k=0; k<hidnum; k++){
                double out = neurons[1][k].getValue();
                result[k] = out * (1-out) * (sum);
            }
        }*/
        
        
        return result;
    }
    
    public void determineOutput(Instance instance){
        class_output_value = countOutput(instance);
    }
    
    public double countOutput(Instance instance){
       // System.out.println("Count output invoked");
        setInputLayer(instance.toDoubleArray());
        
        double[] result = new double[instance.numClasses()];
        double finale_result = -1.0;
        if (hidden_layers == 0){
            //Masukkan input ke input layer
            for (int i=0; i<instance.numAttributes()-1;i++){
                neurons[0][i].setValue(instance.value(i));
            }
            //Langsung hitung output
            for (int k = 0 ; k<instance.numClasses() ; k++){
                neurons[1][k].setValue(neurons[1][k].activate(instance));
                
            }
            
            Neuron max = neurons[1][0]; //Initialize
            //Cari nilai maksimal, karena kelas hanya bisa satu, sehingga kelas lain 
            //nilainya 0, sementara kelas maksimal diberi nilai 1
            for (int k = 1; k < instance.numClasses(); k++){
                if (max.getValue() < neurons[1][k].getValue()){
                    max = neurons[1][k];
                }
            }
            for (int k = 0; k < instance.numClasses(); k++){
                if (!neurons[1][k].equals(max)){
                    neurons[1][k].setOutputValue(0);
                } else {
                    neurons[1][k].setOutputValue(1);
                }
                
              //  System.out.println("value 1 "+k+" : "+neurons[1][k].getValue());
                result[k] = neurons[1][k].getValue();
            }
        }
        else if (hidden_layers == 1){
            //Hitung hidden layer
            for (int i=0 ; i<neurons[1].length ; i++){
                neurons[1][i].setValue(neurons[1][i].activate(instance));

            }
            //Hitung output layer
            for (int k = 0 ; k<instance.numClasses() ; k++){                 
                double[] inp = new double[neurons[1].length];
                for (int i=0; i < neurons[1].length ; i++){
                    inp[i] = neurons[1][i].getValue();
                }
                
                neurons[2][k].setValue(neurons[2][k].activate(inp));
            }
            
            
            
        }
        else{
           throw new RuntimeException("Illegal n_hidden_layer");
        }
        
        double max = neurons[neurons.length-1][0].getValue(); //Initialize
            //Cari nilai maksimal, karena kelas hanya bisa satu, sehingga kelas lain 
            //nilainya 0, sementara kelas maksimal diberi nilai 1
            for (int k = 1; k < instance.numClasses(); k++){
                if (max < neurons[neurons.length-1][k].getValue()){
                    max = neurons[neurons.length-1][k].getValue();
                }
            }
            for (int k = 0; k < instance.numClasses(); k++){
                if (max != neurons[neurons.length-1][k].getValue()){
                    neurons[neurons.length-1][k].setOutputValue(0);
                } else {
                    neurons[neurons.length-1][k].setOutputValue(1);
                }
                result[k] = neurons[neurons.length-1][k].getOutputValue();
            }
        //Deciding class
        //If final_result = array result element which contains 1
        for (int i = 0; i<instance.numClasses(); i++){
            if (result[i] == 1.0){
                finale_result = i;
                break;
            }
        }
        
        return finale_result;
    }
    
    public void printAllWeights(){
        for (int i=1 ; i<neurons.length; i++){
            for(int j=0; j<neurons[i].length; j++){ //Banyak neuron pada layer sebelumnya
                System.out.println("Weight from layer "+(i-1)+" to neuron "+i+" "+j+"  :"+neurons[i][j].getWeights().toString());
            }
        }
    }

    private int getNumOutputNeurons() {
        if(neurons==null)
        {
            return 0;
        }

        return neurons[neurons.length-1].length;
    }

    public double getClassOutputValues(){
        return class_output_value;
    }

    public Neuron[][] getNeurons() {
        return neurons;
    }
    
    public double getSumError(){
        return sumError;
    }
    
    public Instances getInstances(){
        return instances;
    }

    public void setOrigin(Instances i){
        origin = new Instances(i);
    }
    
    public Instances getOrigin(){
        return origin;
    }
}
