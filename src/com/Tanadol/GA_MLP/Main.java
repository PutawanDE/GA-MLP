package com.Tanadol.GA_MLP;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        int k = 10;
        String path = "D:\\PUTAWAN\\ComputerProjects\\CI\\HW3-GA\\Data\\z-score_norm\\";

        try {
            int test_tp = 0, test_tn = 0, test_fp = 0, test_fn = 0;
            StringBuilder evalResultStr = new StringBuilder();

            for (int i = 1; i <= k; i++) {
                Pair<double[][], double[][]> trainingSet = readTrainingData(path + "train" + i + ".csv");
                Individual sol = train(trainingSet);

                Pair<double[][], double[][]> testSet = readTrainingData(path + "test" + i + ".csv");
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

    private static Individual train(Pair<double[][], double[][]> inOut) {
        GA_MLP ga = new GA_MLP();
        int rows = inOut.x.length;
        Individual[] lastSol = null;
        Individual bestSolution;
        long startAll = System.currentTimeMillis();

        for (int e = 1; e <= 1; e++) {
            for (int i = 0; i < rows; i++) {
                long start = System.currentTimeMillis();
                double[] input = inOut.x[i];
                double[] desiredOutput = inOut.y[i];

                Individual[] solution = ga.run(150, 50, 0.38, 3,
                        0.005, 0.0, 10.0, input, desiredOutput, lastSol);

                lastSol = solution;

                System.out.println(solution[0].fitness);
                long finish = System.currentTimeMillis();
                System.out.println("Epoch: " + e + " Elapsed Time iteration " + (i + 1) + ": " + (finish - start));
                System.out.println("---------------------------------");
            }
        }

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
}
