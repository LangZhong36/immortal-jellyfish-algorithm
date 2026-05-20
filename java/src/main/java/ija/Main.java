package ija;

import java.util.function.Function;

/**
 * IJA demo: runs all 8 benchmark functions and prints a summary table.
 *
 * Build and run:
 * <pre>
 *   cd immortal-jellyfish-algorithm/java
 *   mvn compile
 *   mvn exec:java -Dexec.mainClass="ija.Main"
 * </pre>
 * Or without Maven:
 * <pre>
 *   javac -d out src/main/java/ija/*.java
 *   java -cp out ija.Main
 * </pre>
 */
public class Main {

    private static final double[][] BOUNDS = {
        {-100.0,  100.0},
        { -30.0,   30.0},
        {  -5.12,   5.12},
        { -32.768, 32.768},
        {-600.0,  600.0},
        { -10.0,   10.0},
        {-500.0,  500.0},
        {  -5.0,   10.0},
    };

    public static void main(String[] args) {
        int dim     = 30;
        int n       = 50;
        int maxIter = 500;

        System.out.printf("%-14s  %16s  %12s  %10s%n",
                          "Function", "Best Fitness", "NFev", "Time (ms)");
        System.out.println("-".repeat(58));

        for (int fid = 0; fid < BenchmarkFunctions.NAMES.length; fid++) {
            final int id  = fid;
            final double lb = BOUNDS[fid][0];
            final double ub = BOUNDS[fid][1];

            Function<double[], Double> objective = x -> BenchmarkFunctions.evaluate(id, x);

            IJA solver = new IJA.Builder()
                .n(n)
                .maxIter(maxIter)
                .seed(42L)
                .build();

            long t0     = System.currentTimeMillis();
            OptimizeResult result = solver.optimize(objective, dim, lb, ub);
            long elapsed = System.currentTimeMillis() - t0;

            System.out.printf("%-14s  %16.6e  %12d  %10d%n",
                              BenchmarkFunctions.NAMES[fid],
                              result.bestFitness,
                              result.nFunctionEvals,
                              elapsed);
        }
    }
}
