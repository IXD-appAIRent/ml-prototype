package org.codingixd.appairend.RandomForestDemo;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.core.Debug;

import java.util.Random;

public class RandomForestTest {

    public static void main(String[] args) throws Exception{

        String inputPath = "./data/pm10.arff";
        // Read Data
        Instances ds = null;

        try {
            ds = DataSource.read(inputPath);
        }catch(Exception e){
            System.err.println(e);
            System.exit(0);
        }

        if (ds.classIndex() == -1) {
            ds.setClassIndex(ds.numAttributes() - 1);
        }

        Filter filter = new Normalize();

        // Build classifier

        int trainSize = (int) Math.round(ds.numInstances() * 0.8);
        int testSize = ds.numInstances() - trainSize;

        ds.randomize(new Debug.Random(1));

        filter.setInputFormat(ds);
        Instances datasetnor = Filter.useFilter(ds, filter);

        Instances traindataset = new Instances(datasetnor, 0, trainSize);
        Instances testdataset = new Instances(datasetnor, trainSize, testSize);

        RandomForest rf = new RandomForest();
        rf.buildClassifier(traindataset);

        // evaluate model
        Evaluation eval = new Evaluation(traindataset);
        eval.evaluateModel(rf, testdataset);

        System.out.println(eval.toSummaryString());

    }
}
