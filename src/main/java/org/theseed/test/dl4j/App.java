package org.theseed.test.dl4j;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.dropout.GaussianDropout;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.dl4j.TabbedTrainingSetReader;


/**
 * Run a deep learning model
 *
 */
public class App
{

    private static Logger log = LoggerFactory.getLogger(App.class);


    public static void main( String[] args )
    {
        try {
            int labelIndex = 4;     //5 values in each row of the iris.txt CSV: 4 input features followed by an
                                    // integer label (class) index. Labels are the 5th value (index 4) in each row
            int numClasses = 3;     // 3 classes (types of iris flowers) in the iris data set.
                                    // Classes have integer values 0, 1 or 2.
            int batchSize = 40;     // Iris data set: 150 examples total.
            int seed = 123;			// constant seed for repeatability
            int layerWidth = 3;		// number of nodes in the middle layer
            int iterations = 600;	// number of iterations per batch
            File inFile = new File("src/test", "iris.tbl");
            List<String> labels = Arrays.asList("setosa", "versicolor", "virginica");
            TabbedTrainingSetReader reader = new TabbedTrainingSetReader(inFile, "species", labels)
                    .setBatchSize(batchSize);
            // Configuration.
            MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .biasUpdater(new Sgd(0.1))
                .updater(new Adam())
                .list()
                    .layer(0, new DenseLayer.Builder().nIn(labelIndex).nOut(numClasses).build())
                    .layer(1, new DenseLayer.Builder().nIn(numClasses).nOut(layerWidth)
                            .dropOut(new GaussianDropout(0.6)).build())
                    .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                    .activation(Activation.SOFTMAX)
                                    .nIn(layerWidth).nOut(numClasses).build())
                .build();
            // First batch is test data, and is also used to normalize.
            DataSet testData = reader.next();
            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(testData);
            normalizer.transform(testData);
            reader.setNormalizer(normalizer);
            //run the model
            MultiLayerNetwork model = new MultiLayerNetwork(configuration);
            model.init();
            model.setListeners(new ScoreIterationListener(500));
            // Train the model
            for (DataSet trainingData : reader) {
                for(int i=0; i < iterations; i++ ) {
                    model.fit(trainingData);
                }
            }
            // HERE IS WHERE WE ACTUALLY USE THE MODEL TO PREDICT
            INDArray output = model.output(testData.getFeatures());
            //evaluate the model on the test set: compare the output to the actual
            Evaluation eval = new Evaluation(numClasses);
            eval.eval(testData.getLabels(), output);
            log.info(eval.stats());
        } catch (IOException e) {
            e.printStackTrace(System.err);
        }
    }
}
