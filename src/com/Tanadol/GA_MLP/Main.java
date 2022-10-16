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

            Individual[] solution = ga.run(100, 100, 0.3, 3,
                    0.001, -1.0, 1.0, inOut.x[0], inOut.y[0]);
            System.out.println(solution[0].fitness);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static Pair<double[][], double[][]> readTrainingData() throws IOException {
        List<double[]> input = new ArrayList<>();
        List<double[]> output = new ArrayList<>();

        BufferedReader br = new BufferedReader(new
                FileReader("D:\\PUTAWAN\\ComputerProjects\\CI\\HW3-GA\\wbc_data.csv"));
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
