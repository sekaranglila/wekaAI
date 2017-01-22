/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package NaiveBayes;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.Serializable;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Scanner;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Utils;
/**
 *
 * @author UX303L
 */


public class NaiveBayes13514004 implements Classifier, Serializable {
    //Kamus Global
    public Instances origin;
    public int [][][] data; //Matriks data frekuensi
    public double [][][] prob; //Matiks data probabilitas
    public int [] kelasdata;
    public double [] kelasprob;
    int numClass, numAtt, numDis; 
    
    //Reader
    public Instances readFile(String filename) throws Exception {
        //Kamus Lokal
        Instances data = null;
        BufferedReader reader;

        //Algoritma
        reader = new BufferedReader(new FileReader(filename));
        data = new Instances(reader);
        data.setClassIndex(data.numAttributes() - 1);
        reader.close();

        return data;
    }

    //Discretize
    public Instances filterData(Instances data) throws Exception {
        //Kamus Lokal
        Discretize filter = new Discretize();
        Instances filterRes;

        //Algoritma
        filter.setInputFormat(data);
        filterRes = Filter.useFilter(data, filter);

        return filterRes;
    }
    
    @Override
    public void buildClassifier(Instances i) {
        //Algoritma
        origin = new Instances(i);
        //Menghitung jumlah attribute dan kelas
        numAtt = i.numAttributes()-1;
        numClass = i.numClasses();
        
        //Inisialisasi matrix 3 dimensi
        data = new int [numAtt][numClass][0];
        prob = new double [numAtt][numClass][0];
        kelasdata = new int [numClass];
        kelasprob = new double[numClass];
        
        Enumeration<Instance> enu1 = i.enumerateInstances();
        while (enu1.hasMoreElements()){
            Instance ins = enu1.nextElement();
            Enumeration<Attribute> enu_t = i.enumerateAttributes();
            int x = 0;
            while (enu_t.hasMoreElements()){
                Attribute att = enu_t.nextElement();
                numDis = att.numValues();
                data[x][(int)ins.classValue()] = new int [numDis];
                prob[x][(int)ins.classValue()] = new double [numDis];
                x++;
            }
        }
        
        //Mengisi matriks Frekuensi
        Enumeration<Instance> enu2 = i.enumerateInstances();
        while (enu2.hasMoreElements()){
            Instance ins = enu2.nextElement();
            Enumeration<Attribute> enu_t = i.enumerateAttributes();
            int x = 0;
            while (enu_t.hasMoreElements()){
                Attribute att = enu_t.nextElement();
                data[x][(int)ins.classValue()][(int)ins.value(att)]++;
                x++;
            }
            kelasdata[(int)ins.classValue()]++;
        }
        
        //Menghitung probabilitas kelas
        double numInstances = (double)i.numInstances();
        for (int y = 0; y < numClass; y++){
            kelasprob[y] = (double)kelasdata[y]/numInstances;
        }
        
        //Mengisi matriks probabilitas
        Enumeration<Instance> enu3 = i.enumerateInstances();
        while (enu3.hasMoreElements()){
            Instance ins = enu3.nextElement();
            Enumeration<Attribute> enu_t = i.enumerateAttributes();
            int x = 0;
            while (enu_t.hasMoreElements()){
                Attribute att = enu_t.nextElement();
                int sumDis = Utils.sum(data[x][(int)ins.classValue()]);
                numDis = att.numValues();
                for (int z = 0; z < numDis; z++){
                    int y = (int)ins.classValue();
                    prob[x][y][z] = ((double)data[x][y][z]/(double)sumDis);
                }
                x++;
            }
        }
        
    }

    @Override
    public double classifyInstance(Instance i) throws Exception{
        //Kamus Lokal
        double max;
        double [] result = new double [numClass];
        
        //Algoritma
        System.arraycopy(distributionForInstance(i), 0, result, 0, numClass);
        max = result[0];
        for (int x = 1; x < numClass; x++){
            if (max < result[x]){
                max = result[x];
            }
        }
        return max;
    }

    @Override
    public double[] distributionForInstance(Instance instnc) throws Exception {
        Attribute ins;
        double [] result = new double [numClass];
        
        //Algoritma
        //Menghitung probabilitas
        for (int j = 0; j < numClass; j++){
            result[j] = kelasprob[j];
            Enumeration<Attribute> e2 = instnc.enumerateAttributes();
            int i = 0;
            while (e2.hasMoreElements()){
                ins = e2.nextElement();
                result[j] = result[j] * prob[i][j][(int)instnc.value(ins)];
                i++;
            }       
        }
    
        return result;
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable( Capabilities.Capability.MISSING_VALUES );

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);

        return result; 
    }
    
}