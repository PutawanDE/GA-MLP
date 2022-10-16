package com.Tanadol.GA_MLP;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class GA_MLP {
    private static final Random random = new Random();
    private final int populationSize = 50;

    private Individual[] population = new Individual[populationSize];
    private double totalFitness;

    private double[] input = new double[30];
    private double[] desireOutput = new double[1];

    public void run(int maxGeneration) {
        initPopulation();

        int gen = 0;
        while (gen < maxGeneration) {
            evaluateFitness();
            select();
            crossOver();
            mutation();
            gen++;
        }
    }

    private void initPopulation() {
        for (int i = 0; i < population.length; i++) {
            population[i] = new Individual();
        }
    }

    private void evaluateFitness() {
        totalFitness = 0;
        for (Individual p : population) {
            p.fitness = 1.0 - p.network.feedForward(input, desireOutput);
            totalFitness += p.fitness;
        }
    }

    private void select() {
        List<Individual> newPopulation = new ArrayList<>(populationSize);
        // Using Roulette Wheel Selection
        // Find selection prob. and init cumulative prob.
        int n = population.length;
        double[] cumulativeProb = new double[n];
        for (int i = 0; i < n; i++) {
            population[i].selectProb = population[i].fitness / totalFitness;
            if (i == 0) {
                cumulativeProb[i] = population[i].selectProb;
            } else {
                cumulativeProb[i] = population[i].selectProb + cumulativeProb[i - 1];
            }
        }

        // Apply selection operator n times
        for (int i = 0; i < n; i++) {
            double rand = random.nextDouble();

            if (0.0 <= rand || rand <= cumulativeProb[0]) {
                newPopulation.add(population[0]);
            } else {
                for (int j = 1; j < n; j++) {
                    if (cumulativeProb[j - 1] <= rand || rand <= cumulativeProb[j]) {
                        newPopulation.add(population[j]);
                        break;
                    }
                }
            }
        }

        // Selection for replacement
        int currentSize = newPopulation.size();
        while (currentSize < populationSize) {
            int randIdx = random.nextInt(populationSize);
            newPopulation.add(population[randIdx]);
            currentSize++;
        }

        population = newPopulation.toArray(new Individual[populationSize]);
    }

    private void crossOver() {
        // TODO: Cross Chromosome, Update weights
    }

    private void mutation() {
        // TODO: Mutate chromosome, Update weights
    }
}
