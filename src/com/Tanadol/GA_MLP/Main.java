package com.Tanadol.GA_MLP;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        try {
            GA_MLP ga = new GA_MLP();
            Pair<double[][], double[][]> inOut = readTrainingData();
            int rows = inOut.x.length;

            Individual[] lastSol = null;
            for (int i = 0; i < rows; i++) {
                double[] inputVect = inOut.x[i];
                double[] outputVect = inOut.y[i];

                Individual[] solution = ga.run(100, 200, 0.38, 3,
                        0.001, -2.0, 2.0, inputVect, outputVect, lastSol);

                lastSol = solution;
                double maxFitness = Integer.MIN_VALUE;
                for (Individual individual : solution) {
                    if (individual.fitness > maxFitness) maxFitness = individual.fitness;
                }
                System.out.println("D: " + maxFitness);
                System.out.println("---------------------------------");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static Pair<double[][], double[][]> readTrainingData() throws IOException {
        List<double[]> input = new ArrayList<>();
        List<double[]> output = new ArrayList<>();

        BufferedReader br = new BufferedReader(new
                FileReader("D:\\PUTAWAN\\ComputerProjects\\CI\\HW3-GA\\Data\\z-score_norm\\train1.csv"));
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
