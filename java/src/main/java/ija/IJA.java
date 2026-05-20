package ija;

import java.util.*;
import java.util.function.Function;

/**
 * Immortal Jellyfish Algorithm (IJA) — Java implementation.
 *
 * <p>Inspired by the lifecycle of <em>Turritopsis dohrnii</em> and the
 * inverse-square light-intensity model of a physical lighthouse.</p>
 *
 * <p>Usage example:
 * <pre>{@code
 * IJA solver = new IJA.Builder().n(50).maxIter(500).seed(42).build();
 * OptimizeResult result = solver.optimize(x -> Arrays.stream(x).map(v -> v*v).sum(),
 *                                          30, -100.0, 100.0);
 * System.out.println(result.bestFitness);
 * }</pre>
 * </p>
 */
public class IJA {

    private final int    n;
    private final int    maxIter;
    private final int    topK;
    private final double levyBeta;
    private final int    archiveSize;
    private final int    ageMax;
    private final double decayLambda;
    private final double alphaMax;
    private final double alphaMin;
    private final double diversityThreshold;
    private final long   seed;

    private Random rng;

    private IJA(Builder b) {
        this.n                  = b.n;
        this.maxIter            = b.maxIter;
        this.topK               = b.topK;
        this.levyBeta           = b.levyBeta;
        this.archiveSize        = b.archiveSize;
        this.ageMax             = b.ageMax;
        this.decayLambda        = b.decayLambda;
        this.alphaMax           = b.alphaMax;
        this.alphaMin           = b.alphaMin;
        this.diversityThreshold = b.diversityThreshold;
        this.seed               = b.seed;
    }

    public OptimizeResult optimize(Function<double[], Double> objective,
                                   int dim, double lb, double ub) {
        return optimize(objective, dim,
                        uniformArray(dim, lb), uniformArray(dim, ub));
    }

    public OptimizeResult optimize(Function<double[], Double> objective,
                                   int dim, double[] lbArr, double[] ubArr) {
        rng = (seed >= 0) ? new Random(seed) : new Random();

        double[][] population = initialisePopulation(dim, lbArr, ubArr);
        double[]   fitnesses  = evaluateAll(objective, population);
        int[]      ages       = new int[n];
        double[]   phases     = new double[n];
        for (int i = 0; i < n; i++) phases[i] = rng.nextDouble() * 2.0 * Math.PI;

        EliteArchive archive  = new EliteArchive(archiveSize);
        archive.update(population, fitnesses);

        int    bestIdx = argmin(fitnesses);
        double bestFit = fitnesses[bestIdx];
        double[] bestPos = Arrays.copyOf(population[bestIdx], dim);

        List<Double> curve = new ArrayList<>();
        curve.add(bestFit);
        long nfev = n;

        for (int iter = 1; iter <= maxIter; iter++) {
            double tau   = (double) iter / maxIter;
            int    phase = getPhase(tau);
            double alpha = adaptiveAlpha(tau);
            int    nEph  = adaptiveNEphyra(tau);

            int[] eliteIdx = topKIndices(fitnesses, Math.min(topK, n));

            for (int i = 0; i < n; i++) {
                double[] xi = population[i];
                double   fi = fitnesses[i];

                if (phase == 0) {
                    double[] stepped = levyExploration(xi, bestPos, lbArr, ubArr);
                    double   newFit  = objective.apply(stepped);
                    nfev++;
                    if (newFit < fi) { population[i] = stepped; fitnesses[i] = newFit; ages[i] = 0; }
                    else ages[i]++;

                } else if (phase == 1) {
                    double[][] children  = strobilate(xi, lbArr, ubArr, nEph);
                    nfev += nEph;
                    double bestChildFit = fi;
                    double[] bestChild  = xi;
                    for (double[] child : children) {
                        double cf = objective.apply(child);
                        nfev++;
                        if (cf < bestChildFit) { bestChildFit = cf; bestChild = child; }
                    }
                    if (bestChildFit < fi) { population[i] = bestChild; fitnesses[i] = bestChildFit; ages[i] = 0; }
                    else ages[i]++;

                } else if (phase == 2) {
                    double[] gs    = population[eliteIdx[0]];
                    double   distSq = distSq(xi, gs);
                    double   inten  = lighthouseIntensity(distSq, tau);
                    double[] moved  = lighthouseAttract(xi, gs, inten, alpha, lbArr, ubArr);
                    moved = pulseSwim(moved, tau, phases[i], lbArr, ubArr);

                    if (ages[i] >= ageMax) {
                        double[] result = transdifferentiate(moved, lbArr, ubArr);
                        double   nf     = objective.apply(result);
                        nfev += 2;
                        if (nf < fi) { population[i] = result; fitnesses[i] = nf; }
                        ages[i] = 0;
                    } else {
                        double nf = objective.apply(moved);
                        nfev++;
                        if (nf < fi) { population[i] = moved; fitnesses[i] = nf; ages[i] = 0; }
                        else ages[i]++;
                    }
                    double[] archMember = archive.sample();
                    if (archMember != null) {
                        double[] bud    = archiveBud(population[i], archMember, lbArr, ubArr);
                        double   budFit = objective.apply(bud);
                        nfev++;
                        if (budFit < fitnesses[i]) { population[i] = bud; fitnesses[i] = budFit; }
                    }

                } else {
                    double spread = (ubArr[0] - lbArr[0]) * 0.05 * (1 - tau) * (1 - tau);
                    double[] candidate = new double[dim];
                    for (int j = 0; j < dim; j++)
                        candidate[j] = clamp(bestPos[j] + spread * rng.nextGaussian(), lbArr[j], ubArr[j]);
                    double cf = objective.apply(candidate);
                    nfev++;
                    if (cf < fitnesses[i]) { population[i] = candidate; fitnesses[i] = cf; }
                    double[] archMember = archive.sample();
                    if (archMember != null) {
                        double[] bud    = archiveBud(population[i], archMember, lbArr, ubArr);
                        double   budFit = objective.apply(bud);
                        nfev++;
                        if (budFit < fitnesses[i]) { population[i] = bud; fitnesses[i] = budFit; }
                    }
                }
            }

            diversityGuard(population, fitnesses, objective, lbArr, ubArr);
            archive.update(population, fitnesses);

            int curBest = argmin(fitnesses);
            if (fitnesses[curBest] < bestFit) {
                bestFit = fitnesses[curBest];
                bestPos = Arrays.copyOf(population[curBest], dim);
            }
            curve.add(bestFit);
        }

        return new OptimizeResult(bestFit, bestPos, curve, nfev, maxIter);
    }

    private double[][] initialisePopulation(int dim, double[] lb, double[] ub) {
        double[][] pop = new double[n][dim];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < dim; j++)
                pop[i][j] = lb[j] + rng.nextDouble() * (ub[j] - lb[j]);
        return pop;
    }

    private double[] evaluateAll(Function<double[], Double> obj, double[][] pop) {
        double[] fits = new double[pop.length];
        for (int i = 0; i < pop.length; i++) fits[i] = obj.apply(pop[i]);
        return fits;
    }

    private double[] levyExploration(double[] xi, double[] best, double[] lb, double[] ub) {
        int dim = xi.length;
        double[] step  = levySamples(dim);
        double   scale = 0.01 * (ub[0] - lb[0]);
        double[] result = new double[dim];
        for (int j = 0; j < dim; j++)
            result[j] = clamp(xi[j] + scale * step[j] * (xi[j] - best[j]), lb[j], ub[j]);
        return result;
    }

    private double[][] strobilate(double[] xi, double[] lb, double[] ub, int nEph) {
        int dim = xi.length;
        double[] scale = new double[dim];
        for (int j = 0; j < dim; j++) scale[j] = (ub[j] - lb[j]) / 6.0;
        double[][] children = new double[nEph][dim];
        for (int k = 0; k < nEph; k++) {
            double[] levy = levySamples(dim);
            for (int j = 0; j < dim; j++)
                children[k][j] = clamp(xi[j] + rng.nextGaussian() * levy[j] * scale[j], lb[j], ub[j]);
        }
        return children;
    }

    private double[] lighthouseAttract(double[] xi, double[] gs, double intensity,
                                        double alpha, double[] lb, double[] ub) {
        int dim = xi.length;
        double dist = Math.sqrt(distSq(xi, gs));
        double[] result = new double[dim];
        for (int j = 0; j < dim; j++) {
            double noise = 0.01 * rng.nextGaussian() * dist;
            result[j] = clamp(xi[j] + alpha * intensity * (gs[j] - xi[j]) + noise, lb[j], ub[j]);
        }
        return result;
    }

    private double[] pulseSwim(double[] xi, double tau, double phase, double[] lb, double[] ub) {
        int dim = xi.length;
        double amp = 0.1 * (1 - tau) * (1 - tau);
        double osc = amp * Math.sin(2 * Math.PI * 0.5 * tau + phase);
        double[] result = new double[dim];
        for (int j = 0; j < dim; j++)
            result[j] = clamp(xi[j] + osc * (ub[j] - lb[j]), lb[j], ub[j]);
        return result;
    }

    private double[] transdifferentiate(double[] xi, double[] lb, double[] ub) {
        int dim = xi.length;
        double[] opposite = new double[dim];
        for (int j = 0; j < dim; j++)
            opposite[j] = clamp(lb[j] + ub[j] - xi[j], lb[j], ub[j]);
        return opposite;
    }

    private double[] archiveBud(double[] xi, double[] archMember, double[] lb, double[] ub) {
        int dim = xi.length;
        double[] levy   = levySamples(dim);
        double[] result = new double[dim];
        for (int j = 0; j < dim; j++)
            result[j] = clamp(xi[j] + levy[j] * (xi[j] - archMember[j]), lb[j], ub[j]);
        return result;
    }

    private void diversityGuard(double[][] pop, double[] fits,
                                 Function<double[], Double> obj,
                                 double[] lb, double[] ub) {
        int dim = pop[0].length;
        double[] mean = new double[dim];
        for (double[] p : pop)
            for (int j = 0; j < dim; j++) mean[j] += p[j];
        for (int j = 0; j < dim; j++) mean[j] /= n;
        double spread = ub[0] - lb[0];
        if (spread <= 0) spread = 1.0;
        double div = 0;
        for (double[] p : pop)
            for (int j = 0; j < dim; j++) div += Math.abs(p[j] - mean[j]) / spread;
        div /= (double)(n * dim);
        if (div >= diversityThreshold) return;
        int nReset = Math.max(1, (int)(n * 0.2));
        Integer[] order = sortedIndicesDescending(fits);
        for (int k = 0; k < nReset; k++) {
            int idx = order[k];
            for (int j = 0; j < dim; j++) pop[idx][j] = lb[j] + rng.nextDouble() * (ub[j] - lb[j]);
            fits[idx] = obj.apply(pop[idx]);
        }
    }

    private double[] levySamples(int size) {
        double beta   = levyBeta;
        double num    = gamma(1 + beta) * Math.sin(Math.PI * beta / 2);
        double den    = gamma((1 + beta) / 2) * beta * Math.pow(2, (beta - 1) / 2.0);
        double sigmaU = Math.pow(num / den, 1.0 / beta);
        double[] out  = new double[size];
        for (int i = 0; i < size; i++) {
            double u = rng.nextGaussian() * sigmaU;
            double v = rng.nextGaussian();
            out[i] = u / Math.pow(Math.abs(v), 1.0 / beta);
        }
        return out;
    }

    private static double gamma(double x) {
        double[] c = {
            76.18009172947146, -86.50532032941677, 24.01409824083091,
            -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5
        };
        double y = x;
        double tmp = x + 5.5;
        tmp -= (x + 0.5) * Math.log(tmp);
        double ser = 1.000000000190015;
        for (double v : c) ser += v / ++y;
        return Math.exp(-tmp + Math.log(2.5066282746310005 * ser / x));
    }

    private static double lighthouseIntensity(double distSq, double tau) {
        return (1.0 / (1.0 + 0.01 * distSq)) * Math.exp(-0.5 * tau);
    }

    private double adaptiveAlpha(double tau) {
        double w = 1.0 / (1.0 + Math.exp(-12.0 * (tau - 0.5)));
        return alphaMax * (1.0 - w) + alphaMin * w;
    }

    private static int adaptiveNEphyra(double tau) {
        double frac = Math.max(0, Math.min(1, (tau - 0.15) / 0.35));
        return Math.max(2, (int) Math.round(5.0 - frac * 3.0));
    }

    private static int getPhase(double tau) {
        if (tau < 0.15) return 0;
        if (tau < 0.50) return 1;
        if (tau < 0.85) return 2;
        return 3;
    }

    private static double distSq(double[] a, double[] b) {
        double s = 0;
        for (int j = 0; j < a.length; j++) s += (a[j] - b[j]) * (a[j] - b[j]);
        return s;
    }

    private static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    private static int argmin(double[] arr) {
        int idx = 0;
        for (int i = 1; i < arr.length; i++) if (arr[i] < arr[idx]) idx = i;
        return idx;
    }

    private static int[] topKIndices(double[] arr, int k) {
        Integer[] order = new Integer[arr.length];
        for (int i = 0; i < arr.length; i++) order[i] = i;
        Arrays.sort(order, Comparator.comparingDouble(i -> arr[i]));
        int[] result = new int[k];
        for (int i = 0; i < k; i++) result[i] = order[i];
        return result;
    }

    private static Integer[] sortedIndicesDescending(double[] arr) {
        Integer[] order = new Integer[arr.length];
        for (int i = 0; i < arr.length; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Double.compare(arr[b], arr[a]));
        return order;
    }

    private static double[] uniformArray(int size, double val) {
        double[] arr = new double[size];
        Arrays.fill(arr, val);
        return arr;
    }

    public static class Builder {
        int    n                  = 50;
        int    maxIter            = 500;
        int    topK               = 3;
        double levyBeta           = 1.5;
        int    archiveSize        = 15;
        int    ageMax             = 25;
        double decayLambda        = 0.5;
        double alphaMax           = 2.0;
        double alphaMin           = 0.05;
        double diversityThreshold = 0.005;
        long   seed               = -1L;

        public Builder n(int n)                             { this.n = n;                 return this; }
        public Builder maxIter(int v)                       { maxIter = v;                return this; }
        public Builder topK(int v)                          { topK = v;                   return this; }
        public Builder levyBeta(double v)                   { levyBeta = v;               return this; }
        public Builder archiveSize(int v)                   { archiveSize = v;            return this; }
        public Builder ageMax(int v)                        { ageMax = v;                 return this; }
        public Builder decayLambda(double v)                { decayLambda = v;            return this; }
        public Builder alphaMax(double v)                   { alphaMax = v;               return this; }
        public Builder alphaMin(double v)                   { alphaMin = v;               return this; }
        public Builder diversityThreshold(double v)         { diversityThreshold = v;     return this; }
        public Builder seed(long v)                         { this.seed = v;              return this; }
        public IJA build()                                  { return new IJA(this); }
    }
}
