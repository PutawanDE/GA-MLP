package com.Tanadol.GA_MLP;

import java.util.ArrayList;
import java.util.List;

public class Individual {
    private static final MathFunction leakyReluFn = (x) -> {
        if (x <= 0) return 0.01 * x;
        else return x;
    };

    private static final MathFunction sigmoidFn = (x) -> 1.0 / (1.0 + Math.exp(x));

    private static final double minWeight = 0.0;
    private static final double maxWeight = 1.0;
    private static final int[] nodes = new int[]{30, 16, 1};

    protected Network network;
    protected List<Double> chromosome = new ArrayList<>();
    protected double fitness;
    protected double selectProb;

    public Individual() {
        Matrix[] biases = new Matrix[nodes.length - 1];
        for (int i = 0; i < nodes.length - 1; i++) {
            biases[i] = new Matrix(nodes[i + 1], 1);
        }

        network = new Network(nodes, leakyReluFn, sigmoidFn, minWeight, maxWeight, biases);
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
}
