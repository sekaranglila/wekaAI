package FeedForwardNeuralNetwork;

import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.instance.Randomize;

public class FeedForwardNeuralNetwork extends AbstractClassifier implements java.io.Serializable
{
    private FeedForwardNeuralNetworkAlgorithm FFNN;
    private boolean normalized = false;
    
    public FeedForwardNeuralNetworkAlgorithm getAlg(){
        return FFNN;
    }
    
    public void Normalize() throws Exception{
        
    }
    
    public void Randomize() throws Exception{
        Randomize randomize = new Randomize();
        randomize.setInputFormat(FFNN.getInstances());
        buildClassifier(Filter.useFilter(FFNN.getInstances(), randomize));
    }
    
    public void NTB() throws Exception{
        NominalToBinary nominalToBinary = new NominalToBinary();
        nominalToBinary.setInputFormat(FFNN.getInstances());
        buildClassifier(Filter.useFilter(FFNN.getInstances(), nominalToBinary));
    }
    
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        Instances origin = new Instances(instances);
        // can classifier handle the data?
        getCapabilities().testWithFail(instances);
        
        // remove instances with missing class
        instances = new Instances(instances);
        instances.deleteWithMissingClass();
        
        int columnIndex = instances.classIndex() + 1;
        if (instances.classIndex() != instances.numAttributes() -1){
            String order = "";
            for (int i = 1; i < instances.numAttributes() + 1; i++) {
              // skip new class
              if (i == columnIndex)
                continue;

              if (!order.equals(""))
                order += ",";
              order += Integer.toString(i);
            }
            if (!order.equals(""))
              order += ",";
            order += Integer.toString(columnIndex);

            // process data
            Reorder reorder = new Reorder();
            reorder.setAttributeIndices(order);
            System.out.println(order);
            reorder.setInputFormat(instances);
            instances = Filter.useFilter(instances, reorder);
            
            

            // set class index
            instances.setClassIndex(instances.numAttributes() - 1);
        }
        FFNN = new FeedForwardNeuralNetworkAlgorithm(instances);
        
        //NORMALIZE
        Normalize normalize = new Normalize();
        normalize.setInputFormat(instances);
        instances = Filter.useFilter(instances, normalize);
        normalized = true;
        
        //RANDOMIZE
        Randomize randomize = new Randomize();
        randomize.setInputFormat(instances);
        instances = Filter.useFilter(instances, randomize);
        
        FFNN.setOrigin(origin);
        trainModel(instances,1,30);
    }
    
    
    public void trainModel(Instances instances, int hidden_layer, int hidden_neurons){
        //Initialize
        FFNN.buildModel(hidden_layer, hidden_neurons);
        //Start Training
        double[] input = null;
        int j = 1;
        //j itu buat ngatur banyaknya iterasi training
        int error = 0;
        while (FFNN.getSumError() > error && j <= 10000){
            for (int i = 0; i<instances.size(); i++){
                FFNN.clearModel();
                input = instances.get(i).toDoubleArray();
                FFNN.setInputLayer(input);
                FFNN.determineOutput(instances.get(i));
                FFNN.updateModel(instances.get(i));
            }
            j++;
        }
    }
    
    //Ini buat nampilin output array
    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        Instance temp = instance;
        Normalize normalize = new Normalize();
        if (normalized){
            FFNN.getOrigin().add(instance);
            normalize.setInputFormat(FFNN.getOrigin());
            temp = Filter.useFilter(FFNN.getOrigin(), normalize).lastInstance();
            FFNN.getOrigin().remove(FFNN.getOrigin().lastInstance());
        }
        
        FFNN.setInputLayer(temp.toDoubleArray());
        FFNN.determineOutput(temp);
        FFNN.updateModel(temp);
        double[] result = new double[(FFNN.getNeurons())[FFNN.getNeurons().length-1].length];
        for (int i = 0 ; i < result.length ; i++){
            result[i] = (FFNN.getNeurons())[FFNN.getNeurons().length-1][i].getOutputValue();
        }
        
        return result;
        
                
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] array = distributionForInstance(instance);
        double max = array[0];
        int idx = 0;
        for (int i = 1 ; i < array.length ; i++){
            if (array[i] > max){
                max = array[i];
                idx = i;
            }
        } 
         
        return idx;
    }
    
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.NUMERIC_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        return result;
    }
    
    
    
}