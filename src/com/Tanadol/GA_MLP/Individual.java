package com.Tanadol.GA_MLP;

import java.util.ArrayList;
import java.util.List;

public class Individual {
    private static final MathFunction linearFn = (x) -> x;
    private static final MathFunction sigmoidFn = (x) -> 1.0 / (1.0 + Math.exp(x));
    private static final MathFunction leakyReluFn = (x) -> {
        if (x <= 0) return 0.01 * x;
        else return x;
    };

    private static final MathFunction tanhFn = (x) -> 2.0 / (1 + Math.exp(-2.0 * x)) - 1.0;

    private static final double minWeight = 0.0;
    private static final double maxWeight = 1.0;
    private static final int[] nodes = new int[]{30, 1, 1};
    private static final Matrix[] biases = initBiasMat();

    protected Network network;
    protected List<Double> chromosome = new ArrayList<>();
    protected double fitness;
    protected boolean isElite;

    private static Matrix[] initBiasMat() {
        Matrix[] biases = new Matrix[nodes.length - 1];
        for (int i = 0; i < nodes.length - 1; i++) {
            biases[i] = new Matrix(nodes[i + 1], 1);
        }
        return biases;
    }

    public Individual(Individual i) {
        this.network = new Network(i.network);
        this.chromosome = new ArrayList<>(i.chromosome);
        this.fitness = i.fitness;
        this.isElite = i.isElite;
    }

    public Individual() {
        network = new Network(nodes, sigmoidFn, sigmoidFn, minWeight, maxWeight, biases);
        updateChromosome(network.weights);
    }

    protected double evaluateFitness(double[] input, double[] desiredOutput) {
        network = new Network(nodes, sigmoidFn, sigmoidFn, minWeight, maxWeight, biases, chromosome);

        double loss = network.feedForward(input, desiredOutput);
        fitness = 1.0 / (loss + 0.000001);
        updateChromosome(network.weights);
        return fitness;
    }

    private void updateChromosome(Matrix[] weights) {
        chromosome.clear();
        for (Matrix m : weights) {
            for (int i = 0; i < m.getRows(); i++) {
                for (int j = 0; j < m.getCols(); j++) {
                    chromosome.add(m.data[i][j]);
                }
            }
        }
    }

    public int[] evaluateInput(double[][] input, double[][] desiredOutputs, StringBuilder evalStringSb) {
        int tp = 0, tn = 0, fp = 0, fn = 0;
        for (int i = 0; i < input.length; i++) {
            this.network.feedForward(input[i], desiredOutputs[i]);

            double predicted = this.network.activations[nodes.length - 1].data[0][0];

            System.out.println(predicted);

//             1 for positive, 0 for negative, positive->first output is 1
            int predictedPositiveOrNegative = predicted >= 0.4 ? 1 : 0;
            int actualPositiveOrNegative = (int) desiredOutputs[i][0];

            if (actualPositiveOrNegative == 1 && predictedPositiveOrNegative == 1) {
                tp++;
            } else if (actualPositiveOrNegative == 1 && predictedPositiveOrNegative == 0) {
                fn++;
            } else if (actualPositiveOrNegative == 0 && predictedPositiveOrNegative == 1) {
                fp++;
            } else if (actualPositiveOrNegative == 0 && predictedPositiveOrNegative == 0) {
                tn++;
            }
        }

        evalStringSb.append(",actually positive (1),").append("actually negative (0)\n");
        evalStringSb.append("predicted positive (1),").append(tp).append(',').append(fp).append('\n');
        evalStringSb.append("predicted negative (0),").append(fn).append(',').append(tn).append('\n');

        return new int[]{tp, fp, fn, tn};
    }
}
