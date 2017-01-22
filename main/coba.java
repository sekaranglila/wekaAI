/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package main;

import FeedForwardNeuralNetwork.FeedForwardNeuralNetworkAlgorithm;
import FeedForwardNeuralNetwork.Neuron;
import java.io.BufferedReader;
import java.io.FileReader;
import weka.core.Instances;

/**
 *
 * @author user-ari
 */
public class coba {
    public static void main(String[] args) throws Exception {
        BufferedReader breader = null;
        breader = new BufferedReader(new FileReader("src/main/Team.arff"));
        Instances inputTrain = new Instances (breader);
        inputTrain.setClassIndex(inputTrain.numAttributes() -1);
        breader.close();
        FeedForwardNeuralNetworkAlgorithm FFNN = new FeedForwardNeuralNetworkAlgorithm(inputTrain);
        FFNN.buildModel(1,5);
        FFNN.printModel();
        FFNN.printAllWeights();
        
        double[] arr = inputTrain.get(60).toDoubleArray();
        FFNN.setInputLayer(arr);
        FFNN.determineOutput(inputTrain.get(60));
        System.out.println(FFNN.getClassOutputValues());
        FFNN.updateModel(inputTrain.get(60));
        FFNN.printModel();
        FFNN.printAllWeights();
        System.out.println("Class : "+FFNN.getClassOutputValues());
        
        
        System.out.println("\nupdate again!!!!\n");
        FFNN.clearModel();
        arr = null;
        arr = inputTrain.get(61).toDoubleArray();
        FFNN.setInputLayer(arr);
        FFNN.determineOutput(inputTrain.get(61));
        FFNN.updateModel(inputTrain.get(61));
        FFNN.printModel();
        FFNN.printAllWeights();
        System.out.println("Class : "+FFNN.getClassOutputValues());
        
        System.out.println("\nupdate again!!!!\n");
        FFNN.clearModel();
        arr = null;
        arr = inputTrain.get(62).toDoubleArray();
        FFNN.setInputLayer(arr);
        FFNN.determineOutput(inputTrain.get(62));
        FFNN.updateModel(inputTrain.get(62));
        FFNN.printModel();
        FFNN.printAllWeights();
        System.out.println("Class : "+FFNN.getClassOutputValues());
        
    }
    
}
