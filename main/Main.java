/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package main;

import FeedForwardNeuralNetwork.FeedForwardNeuralNetwork;
import NaiveBayes.NaiveBayes13514004;
import java.io.BufferedReader;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.ObjectOutputStream;
import java.util.Random;
import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.instance.Randomize;

/**
 *
 * @author user-ari
 */
public class Main {
    
   static void saveModel(Classifier C, String namaFile) throws Exception {
        //SAVE 
        // serialize model
        String dir = "models//" + namaFile + ".model";
        weka.core.SerializationHelper.write(dir, C);
    }
    
    public static void main(String[] args) throws Exception {
        Scanner scan = new Scanner(System.in);
        BufferedReader breader = null;
        
        //File input
        System.out.print("Masukkan Nama File Input: ");
        String filename = scan.next();
        breader = new BufferedReader(new FileReader("arff//"+ filename +".arff"));
        Instances awal = new Instances (breader);
        breader.close();
        Instances fileTrain = awal;
        //delete index
        System.out.print("Apakah ada atribut yang ingin dihapus? (0: No, 1: Yes)");
        int pil = scan.nextInt();
        if (pil == 1){
            System.out.print("Masukkan index dari atribut yang inging dihapus:");
            int ind = scan.nextInt();
            String str = "";
            if (ind == 26){
                str = "27";
            } else if (ind == 27){
                str ="28";
            }
            Remove remove = new Remove();
            remove.setAttributeIndices(str);
            remove.setInputFormat(awal);
            fileTrain = Filter.useFilter(awal, remove);
        }
        
        
        //Index kelas
        System.out.print("Masukkan index dari atribut yang menjadi kelas "
                + "(-1 untuk last index): ");
        int classIndex = scan.nextInt();
        if (classIndex == -1) classIndex = fileTrain.numAttributes() - 1;
        fileTrain.setClassIndex(classIndex);
        
        //Pilih classifier
        System.out.print("Pilih classifier yang akan digunakan (0: NB, 1: FFNN): ");
        int classifierChoice = scan.nextInt();
        Classifier classifier;
        if (classifierChoice == 0){
            classifier = new NaiveBayes13514004();
        } else {    
            classifier = new FeedForwardNeuralNetwork();
        }
       
        //Evaluate
        Evaluation eval = new Evaluation(fileTrain);
        boolean validasi = false;
        do {
            System.out.println("\n\n=================\n==== OPTION ====");
            System.out.println("1. Full Training Scheme");
            System.out.println("2. 10 Fold Validation Scheme");
            System.out.println("3. Split-Test");
            System.out.println("4. Load");
            System.out.println("5. Create new instance");
            System.out.println("6. Exit");
            System.out.print("Enter your option (1/2/3/4/5/6): ");
            int pilihan = scan.nextInt();
            switch (pilihan) {
                case 1:
                    {
                        if (classifierChoice == 0){
                            Discretize filter = new Discretize();
                            Instances filterRes;

                            //Algoritma
                            filter.setInputFormat(fileTrain);
                            filterRes = Filter.useFilter(fileTrain, filter);
                            
                            classifier.buildClassifier(filterRes);
                            eval.evaluateModel(classifier, filterRes);
                        } else {
                            classifier.buildClassifier(fileTrain);
                            eval.evaluateModel(classifier, fileTrain);
                        }
                        //OUTPUT
                        System.out.println(eval.toSummaryString("=== Stratified cross-validation ===\n" +"=== Summary ===",true));
                        System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ==="));
                        System.out.println(eval.toMatrixString("===Confusion matrix==="));
                        System.out.println(eval.fMeasure(1)+" "+eval.recall(1));
                        System.out.println("\nDo you want to save this model(1/0)? ");
                        int c = scan.nextInt();
                        if (c == 1 ){
                            System.out.print("Please enter your file name (*.model) : ");
                            String infile = scan.next();
                            saveModel(classifier, infile);
                        }
                        else {
                            System.out.print("Model not saved.");
                        }       break;
                    }
                
                case 2:
                    {
                        if (classifierChoice == 0){
                            Discretize filter = new Discretize();
                            Instances filterRes;

                            //Algoritma
                            filter.setInputFormat(fileTrain);
                            filterRes = Filter.useFilter(fileTrain, filter);
                            
                            classifier.buildClassifier(filterRes);
                            eval.crossValidateModel(classifier, filterRes, 10, new Random(1));
                        } else {
                            classifier.buildClassifier(fileTrain);
                            eval.crossValidateModel(classifier, fileTrain, 10, new Random(1));
                        }
                        
                        //OUTPUT
                        System.out.println(eval.toSummaryString("=== 10-fold-cross-validation ===\n",true));
                        System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ==="));
                        System.out.println(eval.toMatrixString("===Confusion matrix==="));
                        System.out.println(eval.fMeasure(1)+" "+eval.recall(1));
                        System.out.println("\nDo you want to save this model(1/0)? ");
                        int c = scan.nextInt();
                        if (c == 1 ){
                            System.out.print("Please enter your file name (*.model) : ");
                            String infile = scan.next();
                            saveModel(classifier, infile);
                        }
                        else {
                            System.out.print("Model not saved.");
                        }       break;
                    }
                case 3: {
                    Discretize filter = new Discretize();
                    Instances filterRes;

                    //Algoritma
                    filter.setInputFormat(fileTrain);
                    filterRes = Filter.useFilter(fileTrain, filter);

                    Randomize rand = new Randomize();
                    rand.setInputFormat(filterRes);
                    filterRes = Filter.useFilter(filterRes, rand);
                    int trainSize = (int) Math.round(filterRes.numInstances() * 66/ 100);
                    int testSize = filterRes.numInstances() - trainSize;
                    Instances train = new Instances(filterRes, 0, trainSize);
                    Instances test = new Instances(filterRes, trainSize, testSize);

                  
                    classifier.buildClassifier(train);
                    Evaluation eval2 = new Evaluation(test);
                    eval2.evaluateModel(classifier, train);
                    System.out.println(eval2.toSummaryString("=== Stratified cross-validation ===\n" +"=== Summary ===",true));
                    System.out.println(eval2.toClassDetailsString("=== Detailed Accuracy By Class ==="));
                    System.out.println(eval2.toMatrixString("===Confusion matrix==="));
                    System.out.println(eval2.fMeasure(1)+" "+eval.recall(1));
                    break; 
                }
                case 4:
                    //LOAD
                    // deserialize model
                    Discretize filter = new Discretize();
                    Instances filterRes;

                    //Algoritma
                    filter.setInputFormat(fileTrain);
                    filterRes = Filter.useFilter(fileTrain, filter);
                    System.out.print("Please enter the file name : ");
                    String namaFile = scan.next();
                    Classifier cls = (Classifier) weka.core.SerializationHelper.read("models//"+ namaFile +".model");
                    
                    eval.evaluateModel(cls, filterRes);
                    System.out.println(eval.toSummaryString("=== Stratified cross-validation ===\n" +"=== Summary ===",true));
                    System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ==="));
                    System.out.println(eval.toMatrixString("===Confusion matrix==="));
                    System.out.println(eval.fMeasure(1)+" "+eval.recall(1));
                    break;
                case 5:
                    System.out.println();
                    //ADD New Instance
                    //Copy attributes from instances
                    DenseInstance buffer = new DenseInstance(fileTrain.firstInstance());
                    //Initialization
                    buffer.setDataset(fileTrain);
                    buffer.setMissing(fileTrain.classIndex());
                    //Input
                    for (int i = 0; i < fileTrain.classIndex(); i++){
                        System.out.print("Enter the value for " + buffer.attribute(i).name() + ": ");
                        double val = scan.nextDouble();
                        buffer.setValue(i, val);
                    }
                    //Classify
                    double res = classifier.classifyInstance(buffer);
                    buffer.setValue(fileTrain.classIndex(), res);
                    fileTrain.add(buffer);
                    System.out.println("Class: " + buffer.stringValue(fileTrain.classIndex()));
                    break;
                case 6:
                    validasi = true;
                    break;
                default:
                    System.out.println("Wrong input!");
                    break;
            }
        } while (!validasi);
    }
}
