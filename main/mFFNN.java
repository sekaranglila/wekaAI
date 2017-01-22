/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package main;

import FeedForwardNeuralNetwork.FeedForwardNeuralNetwork;
import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.ObjectOutputStream;
import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

/**
 *
 * @author user-ari
 */
public class mFFNN {
    
   void saveModel(Classifier C, String namaFile) throws Exception {
        //SAVE 
        // serialize model
        weka.core.SerializationHelper.write(namaFile, C);
    }
    
    public static void main(String[] args) throws Exception {
        mFFNN m = new mFFNN();
        BufferedReader breader = null;
        breader = new BufferedReader(new FileReader("src\\main\\iris.arff"));
        Instances fileTrain = new Instances (breader);
        fileTrain.setClassIndex(fileTrain.numAttributes() - 1);
        System.out.println(fileTrain);
        breader.close();
        System.out.println("mFFNN!!!\n\n");
        FeedForwardNeuralNetwork FFNN = new FeedForwardNeuralNetwork();
        
        Evaluation eval = new Evaluation(fileTrain);
        FFNN.buildClassifier(fileTrain);
       
        eval.evaluateModel(FFNN,fileTrain);
        
        //OUTPUT
        Scanner scan = new Scanner(System.in);
        System.out.println(eval.toSummaryString("=== Stratified cross-validation ===\n" +"=== Summary ===",true));
        System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ==="));
        System.out.println(eval.toMatrixString("===Confusion matrix==="));
        System.out.println(eval.fMeasure(1)+" "+eval.recall(1));
        System.out.println("\nDo you want to save this model(1/0)? ");
        FFNN.distributionForInstance(fileTrain.get(0));
        /* int c = scan.nextInt();
        if (c == 1 ){
             System.out.print("Please enter your file name (*.model) : ");
             String infile = scan.next();
             m.saveModel(FFNN,infile);
        }
        else {
            System.out.print("Model not saved.");
        } */
    }
}
