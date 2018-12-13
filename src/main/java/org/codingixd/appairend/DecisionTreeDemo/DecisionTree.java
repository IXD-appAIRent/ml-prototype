package org.codingixd.appairend.DecisionTreeDemo;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

import java.util.Random;

public class DecisionTree {

    public static void main(String[] args) throws Exception{
        String inputPath = "./data/pm10.arff";

        Instances ds = null;

        try {
            ds = ConverterUtils.DataSource.read(inputPath);
        }catch(Exception e){
            System.err.println(e);
            System.exit(0);
        }

        if (ds.classIndex() == -1) {
            ds.setClassIndex(ds.numAttributes() - 1);
        }

        Filter filter = new Normalize();
        int trainSize = (int) Math.round(ds.numInstances() * 0.8);
        int testSize = ds.numInstances() - trainSize;

        ds.randomize(new Random());

        filter.setInputFormat(ds);
        Instances datasetnor = Filter.useFilter(ds, filter);

        Instances traindataset = new Instances(datasetnor, 0, trainSize);
        Instances testdataset = new Instances(datasetnor, trainSize, testSize);

        J48 tree = new J48();

        tree.buildClassifier(traindataset);

        Evaluation eval = new Evaluation(traindataset);
        eval.evaluateModel(tree, testdataset);

        System.out.println(eval.toSummaryString());

    }
}
