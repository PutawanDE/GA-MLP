package com.Tanadol.GA_MLP;

import java.util.*;

public class GA_MLP {
    private static final Random random = new Random();

    private int populationSize;
    private double crossOverRate;

    private Individual[] population;
    private double totalFitness;

    private double[] input = new double[30];
    private double[] desiredOutput = new double[1];

    public Individual[] run(int maxGeneration, int populationSize, double crossoverRate, int crossoverPoint,
                            double mutationProb, double mutateMin, double mutateMax, double[] input,
                            double[] desiredOutput, Individual[] initPopulation) {
        this.populationSize = populationSize;
        this.crossOverRate = crossoverRate;
        this.input = input;
        this.desiredOutput = desiredOutput;

        if (initPopulation == null) initPopulation();

        int gen = 0;
        while (gen < maxGeneration) {
            evaluateFitness();
            if(gen == 0) {
                double max = 0.0;
                for (Individual p : population) {
                    if (p.fitness > max) max = p.fitness;
                }
                System.out.println("f: " + max);
            }
            select();
            crossover(crossoverPoint);
            mutation(mutationProb, mutateMin, mutateMax);
            gen++;
        }
        evaluateFitness();
        return population;
    }

    private void initPopulation() {
        population = new Individual[populationSize];
        for (int i = 0; i < population.length; i++) {
            population[i] = new Individual();
        }
    }

    private void evaluateFitness() {
        totalFitness = 0;
        for (Individual p : population) {
            p.fitness = 1.0 - p.network.feedForward(input, desiredOutput);
            totalFitness += p.fitness;
        }
    }

    private void select() {
        List<Individual> selected = new ArrayList<>(populationSize);
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
                selected.add(population[0]);
            } else {
                for (int j = 1; j < n; j++) {
                    if (cumulativeProb[j - 1] <= rand || rand <= cumulativeProb[j]) {
                        selected.add(population[j]);
                        break;
                    }
                }
            }
        }

        // Selection for replacement
        int currentSize = selected.size();
        while (currentSize < populationSize) {
            int randIdx = random.nextInt(populationSize);
            selected.add(population[randIdx]);
            currentSize++;
        }

        population = selected.toArray(new Individual[populationSize]);
    }

    // N-point Cross-Over
    private void crossover(int n) {
        List<Individual> newPopulation = new ArrayList<>(populationSize);
        List<Individual> matingPool = new ArrayList<>();
        int lastInMatingPool = 0;
        boolean[] isParent = new boolean[populationSize];

        for (int i = 0; i < populationSize; i++) {
            double rand = random.nextDouble();
            if (rand < crossOverRate) {
                matingPool.add(population[i]);
                lastInMatingPool = i;
                isParent[i] = true;
            }
        }

        int m = matingPool.size();
        if (m % 2 != 0) {
            isParent[lastInMatingPool] = false;
            matingPool.remove(m - 1);
            m--;
        }
        Collections.shuffle(matingPool);

        for (int i = 0; i < m; i += 2) {
            int randParent1 = i;
            int randParent2 = i + 1;

            Individual parent1 = matingPool.get(randParent1);
            Individual parent2 = matingPool.get(randParent2);
            Individual offspring1 = new Individual();
            Individual offspring2 = new Individual();

            int[] crossingSites = new int[n + 2];
            crossingSites[0] = 0;
            crossingSites[n - 1] = parent1.chromosome.size();
            for (int j = 1; j < n - 1; j++) {
                crossingSites[j] = random.nextInt(parent1.chromosome.size());
            }
            Arrays.sort(crossingSites);

            for (int j = 1; j < n; j++) {
                if (j % 2 == 0) {
                    offspring1.chromosome.addAll(parent1.chromosome.subList(crossingSites[j - 1], crossingSites[j]));
                    offspring2.chromosome.addAll(parent2.chromosome.subList(crossingSites[j - 1], crossingSites[j]));
                } else {
                    offspring1.chromosome.addAll(parent2.chromosome.subList(crossingSites[j - 1], crossingSites[j]));
                    offspring2.chromosome.addAll(parent1.chromosome.subList(crossingSites[j - 1], crossingSites[j]));
                }
            }

            offspring1.updateNewWeights();
            offspring2.updateNewWeights();
            newPopulation.add(offspring1);
            newPopulation.add(offspring2);
        }

        for (int i = 0; i < populationSize; i++) {
            if (!isParent[i]) newPopulation.add(population[i]);
        }

        population = newPopulation.toArray(new Individual[0]);
    }

    private void mutation(double mutationProb, double min, double max) {
        for (Individual p : population) {
            for (int i = 0; i < p.chromosome.size(); i++) {
                double q = random.nextDouble();
                if (q < mutationProb) {
                    p.chromosome.set(i, random.nextDouble(min, max));
                    p.updateNewWeights();
                }
            }
        }
    }
}
