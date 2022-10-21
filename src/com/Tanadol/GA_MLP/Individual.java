package com.Tanadol.GA_MLP;

import java.util.ArrayList;
import java.util.List;

public class Individual {
    private static final MathFunction linearFn = (x) -> x;
    private static final MathFunction sigmoidFn = (x) -> 1.0 / (1.0 + Math.exp(x));

    private static final double minWeight = 0.0;
    private static final double maxWeight = 1.0;
    private static final int[] nodes = new int[]{30, 16, 1};

    protected Network network;
    protected List<Double> chromosome = new ArrayList<>();
    protected double fitness;
    protected double selectProb;
    protected boolean isElite;

    public Individual() {
        Matrix[] biases = new Matrix[nodes.length - 1];
        for (int i = 0; i < nodes.length - 1; i++) {
            biases[i] = new Matrix(nodes[i + 1], 1);
        }

        network = new Network(nodes, sigmoidFn, sigmoidFn, minWeight, maxWeight, biases);
        updateChromosome();
    }

    protected void updateNewWeights() {
        Matrix[] newWeights = new Matrix[nodes.length - 1];
        int c = 0;
        for (int l = 0; l < nodes.length - 1; l++) {
            newWeights[l] = new Matrix(nodes[l + 1], nodes[l]);
            for (int i = 0; i < newWeights[l].getRows(); i++) {
                for (int j = 0; j < newWeights[l].getCols(); j++) {
                    newWeights[l].data[i][j] = chromosome.get(c);
                    c++;
                }
            }
        }
        network.weights = newWeights;
        updateChromosome();
    }

    private void updateChromosome() {
        chromosome.clear();
        for (Matrix m : network.weights) {
            for (int i = 0; i < m.getRows(); i++) {
                for (int j = 0; j < m.getCols(); j++) {
                    chromosome.add(m.data[i][j]);
                }
            }
        }
    }

    public int[] evaluate(double[][] input, double[][] desiredOutputs, StringBuilder evalStringSb) {
        int tp = 0, tn = 0, fp = 0, fn = 0;
        for (int i = 0; i < input.length; i++) {
            this.network.feedForward(input[i], desiredOutputs[i]);

            double predicted = this.network.activations[nodes.length - 1].data[0][0];
            System.out.println(predicted);

            // 1 for positive, 0 for negative, positive->first output is 1
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
