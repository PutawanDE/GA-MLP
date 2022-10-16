package com.Tanadol.GA_MLP;

import java.util.Random;

interface MathFunction {
    double run(double x);
}

public class Network {
    protected final int inputLength;
    protected final int desiredOutputLength;

    protected int layerCount;
    private int[] nodeInLayerCount;

    protected Matrix[] activations;
    protected Matrix[] weights;
    private Matrix[] biases;
    protected double loss;

    private final double minWeight;
    private final double maxWeight;

    private final MathFunction hiddenLayerActivationFn;
    private final MathFunction outputLayerActivationFn;

    private static final Random random = new Random();

    public Network(int[] nodeInLayerCount, MathFunction hiddenLayerActivation,
                   MathFunction outputLayerActivation, double minWeight, double maxWeight, Matrix[] biases) {
        this.layerCount = nodeInLayerCount.length;
        this.nodeInLayerCount = nodeInLayerCount;

        this.hiddenLayerActivationFn = hiddenLayerActivation;
        this.outputLayerActivationFn = outputLayerActivation;

        this.minWeight = minWeight;
        this.maxWeight = maxWeight;

        weights = new Matrix[layerCount - 1];
        this.biases = biases.clone();
        activations = new Matrix[layerCount];

        initWeight();

        inputLength = nodeInLayerCount[0];
        desiredOutputLength = nodeInLayerCount[nodeInLayerCount.length - 1];
    }

    private void initWeight() {
        for (int k = 0; k < weights.length; k++) {
            weights[k] = new Matrix(nodeInLayerCount[k + 1], nodeInLayerCount[k]);
            for (int j = 0; j < weights[k].getRows(); j++) {
                for (int i = 0; i < weights[k].getCols(); i++) {
                    weights[k].data[j][i] = random.nextDouble(minWeight, maxWeight);
                }
            }
        }
    }

    protected double feedForward(double[] inputVect, double[] desiredOutputVect) {
        double[][] inputMat = new double[inputVect.length][1];
        for (int i = 0; i < inputVect.length; i++) {
            inputMat[i][0] = inputVect[i];
        }

        activations[0] = new Matrix(inputMat);
        for (int i = 1; i < layerCount; i++) {
            MathFunction activationFn = i == layerCount - 1 ? outputLayerActivationFn : hiddenLayerActivationFn;

            Matrix net = Matrix.multiply(weights[i - 1], activations[i - 1]);
            Matrix output = net.add(biases[i - 1]);
            activations[i] = Matrix.applyFunction(output, activationFn);
        }

        loss = calcLoss(desiredOutputVect);
        return loss;
    }

    // calculate Loss
    private double calcLoss(double[] desiredOutputVect) {
        double loss = 0;
        // use binary cross-entropy
        double y = desiredOutputVect[0];
        double p = activations[layerCount - 1].data[0][0];

        if (y == 0.0) {
            loss = -Math.log(1.0 - p);
        } else if (y == 1.0) {
            loss = -Math.log(p);
        }
        return loss;
    }
}
