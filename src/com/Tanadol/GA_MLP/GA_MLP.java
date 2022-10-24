package com.Tanadol.GA_MLP;

import java.util.*;

public class GA_MLP {
    private static final Random random = new Random();

    private int populationSize;
    private double crossOverRate;
    private double elitePercentage = 10;
    private int elitesCount;

    private Individual[] population;
    private Individual[] elites;

    public Individual[] run(int maxGeneration, int populationSize, double crossoverRate, int crossoverPoint,
                            double mutationProb, double mutateMin, double mutateMax, double[] input,
                            double[] desiredOutput, Individual[] startPopulation) {
        this.populationSize = populationSize;
        this.crossOverRate = crossoverRate;
        this.elitesCount = (int) ((elitePercentage / 100.0) * populationSize);
        this.elites = new Individual[elitesCount];

        if (startPopulation == null) initPopulation();
        else population = startPopulation;

        int gen = 0;
        while (gen < maxGeneration) {
            evaluateFitness(input, desiredOutput);
            findElites();
            Individual[] toCrossMutate = Arrays.copyOfRange(population, elites.length, populationSize);
            rouletteWheelSelect(toCrossMutate);
            crossover(toCrossMutate, crossoverPoint);
            mutation(toCrossMutate, mutationProb, mutateMin, mutateMax);
            gen++;
        }
        evaluateFitness(input, desiredOutput);
        return population;
    }

    private void initPopulation() {
        population = new Individual[populationSize];
        for (int i = 0; i < population.length; i++) {
            population[i] = new Individual();
        }
    }

    private void evaluateFitness(double[] input, double[] desiredOutput) {
        for (int i = 0; i < populationSize; i++) {
            population[i].evaluateFitness(input, desiredOutput);
        }
    }

    private void findElites() {
        Arrays.sort(population, (o1, o2) -> Double.compare(o2.fitness, o1.fitness));
        for (int i = 0; i < elitesCount; i++) {
            elites[i] = population[i];
            population[i].isElite = true;
        }
    }

    private void rouletteWheelSelect(Individual[] group) {
        List<Individual> selected = new ArrayList<>(group.length);
        // Using Roulette Wheel Selection
        // Find selection prob. and init cumulative prob.
        int n = group.length;
        double[] cumulativeProb = new double[n];
        double totalFitness = 0;
        for (Individual p : group) {
            totalFitness += p.fitness;
        }

        double[] selectProb = new double[n];
        for (int i = 0; i < n; i++) {
            selectProb[i] = group[i].fitness / totalFitness;
            if (i == 0) {
                cumulativeProb[i] = selectProb[i];
            } else {
                cumulativeProb[i] = selectProb[i] + cumulativeProb[i - 1];
            }
        }

        // Apply selection operator n times
        for (int i = 0; i < n; i++) {
            double rand = random.nextDouble();

            if (rand <= cumulativeProb[0]) {
                selected.add(new Individual(population[0]));
            } else {
                for (int j = 1; j < n; j++) {
                    if (cumulativeProb[j - 1] <= rand && rand <= cumulativeProb[j]) {
                        selected.add(new Individual(population[j]));
                        break;
                    }
                }
            }
        }

        selected.addAll(List.of(elites));
        population = selected.toArray(new Individual[populationSize]);
    }

    // N-point Cross-Over
    private void crossover(Individual[] group, int n) {
        List<Individual> newPopulation = new ArrayList<>(group.length);
        List<Individual> matingPool = new ArrayList<>(group.length);
        int lastInMatingPool = 0;
        boolean[] isParent = new boolean[group.length];

        for (int i = 0; i < group.length; i++) {
            double rand = random.nextDouble();
            if (rand < crossOverRate) {
                matingPool.add(group[i]);
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

            n = n + 2;
            int[] crossingSites = new int[n];
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

            newPopulation.add(offspring1);
            newPopulation.add(offspring2);
        }

        for (int i = 0; i < group.length; i++) {
            if (!isParent[i]) newPopulation.add(group[i]);
        }

        newPopulation.addAll(List.of(elites));
        this.population = newPopulation.toArray(new Individual[0]);
    }

    private void mutation(Individual[] group, double mutationProb, double min, double max) {
        for (Individual p : group) {
            if (!p.isElite) {
                for (int i = 0; i < p.chromosome.size(); i++) {
                    double q = random.nextDouble();
                    if (q < mutationProb) {
                        p.chromosome.set(i, random.nextDouble(min, max));
                    }
                }
            }
        }
    }
}
