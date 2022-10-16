package com.Tanadol.GA_MLP;

public class Main {
    public static void main(String[] args) {
        GA_MLP ga = new GA_MLP();
        Individual[] solution = ga.run(100, 50, 0.3, 3,
                0.001, -1.0, 1.0);
        System.out.println(solution[0].fitness);
    }
}
