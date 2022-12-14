package com.Tanadol.GA_MLP;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        int k = 10;
        String inputPath = "D:\\PUTAWAN\\ComputerProjects\\CI\\HW3-GA\\Data\\z-score_norm\\";

        try {
            int test_tp = 0, test_tn = 0, test_fp = 0, test_fn = 0;
            StringBuilder evalResultStr = new StringBuilder();

            for (int i = 1; i <= k; i++) {
                Pair<double[][], double[][]> trainingSet = readTrainingData(inputPath + "train" + i + ".csv");
                Individual sol = train(trainingSet, k);

                Pair<double[][], double[][]> testSet = readTrainingData(inputPath + "test" + i + ".csv");
                List<Double> output = new ArrayList<>(testSet.x.length);

                int[] testConfusionMat = sol.evaluateInput(testSet.x, testSet.y, evalResultStr);

                evalResultStr.append("Test Data Confusion Matrix: ").append(i).append('\n');
                test_tp += testConfusionMat[0];
                test_fp += testConfusionMat[1];
                test_fn += testConfusionMat[2];
                test_tn += testConfusionMat[3];

                evalResultStr.append('\n');
            }
            System.out.println(evalResultStr);

            evalResultStr = new StringBuilder();
            evalResultStr.append(test_tn).append(',').append(test_fp).append("],[").append(test_fn).append(',').append(test_tp);
            System.out.println(evalResultStr);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static Individual train(Pair<double[][], double[][]> inOut, int k) throws IOException {
        GA_MLP ga = new GA_MLP();
        int rows = inOut.x.length;
        Individual[] lastSol = null;
        Individual bestSolution;
        long startAll = System.currentTimeMillis();

        double[] loss = new double[rows];
        double[] fitness = new double[rows];
        double[] predicted = new double[rows];
        double[] target = new double[rows];

        for (int i = 0; i < rows; i++) {
            long start = System.currentTimeMillis();
            double[] input = inOut.x[i];
            double[] desiredOutput = inOut.y[i];

            Individual[] solution = ga.run(100, 50, 0.8, 3,
                    0.001, 0.07, 1.0, input, desiredOutput, lastSol);

            lastSol = solution;

            System.out.println(solution[0].fitness);
            loss[i] = solution[0].network.loss;
            fitness[i] = solution[0].fitness;
            predicted[i] = solution[0].network.activations[2].data[0][0];
            target[i] = inOut.y[i][0];

            long finish = System.currentTimeMillis();
            System.out.println("Elapsed Time iteration " + (i + 1) + ": " + (finish - start));
            System.out.println("---------------------------------");
        }
        saveArrayToFile(loss, "D:\\PUTAWAN\\ComputerProjects\\CI\\HW3-GA\\result\\loss" + k + ".csv");
        saveArrayToFile(fitness, "D:\\PUTAWAN\\ComputerProjects\\CI\\HW3-GA\\result\\fitness" + k + ".csv");
        saveArrayToFile(predicted, "D:\\PUTAWAN\\ComputerProjects\\CI\\HW3-GA\\result\\predicted" + k + ".csv");
        saveArrayToFile(target, "D:\\PUTAWAN\\ComputerProjects\\CI\\HW3-GA\\result\\target" + k + ".csv");

        bestSolution = lastSol[0];
        System.out.println("D: " + bestSolution.fitness);
        long finishAll = System.currentTimeMillis();
        System.out.println("Total Elapsed Time: " + (finishAll - startAll));
        return bestSolution;
    }

    private static Pair<double[][], double[][]> readTrainingData(String filename) throws IOException {
        List<double[]> input = new ArrayList<>();
        List<double[]> output = new ArrayList<>();

        BufferedReader br = new BufferedReader(new FileReader(filename));
        String line;
        while ((line = br.readLine()) != null) {
            String[] cols = line.split(",");
            if (cols[1].equals("M")) {
                output.add(new double[]{1.0});
            } else {
                output.add(new double[]{0.0});
            }

            double[] inputVect = new double[30];
            for (int j = 0; j < 30; j++) {
                inputVect[j] = Double.parseDouble(cols[j + 2]);
            }
            input.add(inputVect);
        }

        return new Pair<>(input.toArray(new double[0][0]), output.toArray(new double[0][0]));
    }

    private static void saveArrayToFile(double[] array, String filename) {
        File file = new File(filename);
        try (BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(file))) {
            String arrayStr = Arrays.toString(array);
            bufferedWriter.append(arrayStr.substring(1, arrayStr.length() - 1));
        } catch (IOException exception) {
            exception.printStackTrace();
        }
    }
}
