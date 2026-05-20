package ija;

import java.util.*;

/**
 * Immutable container for IJA optimization results.
 */
public class OptimizeResult {

    public final double       bestFitness;
    public final double[]     bestPosition;
    public final List<Double> convergenceCurve;
    public final long         nFunctionEvals;
    public final int          nIterations;

    public OptimizeResult(double bestFitness, double[] bestPosition,
                           List<Double> convergenceCurve,
                           long nFunctionEvals, int nIterations) {
        this.bestFitness      = bestFitness;
        this.bestPosition     = Arrays.copyOf(bestPosition, bestPosition.length);
        this.convergenceCurve = Collections.unmodifiableList(new ArrayList<>(convergenceCurve));
        this.nFunctionEvals   = nFunctionEvals;
        this.nIterations      = nIterations;
    }

    @Override
    public String toString() {
        return String.format("OptimizeResult(bestFitness=%.6e, nIter=%d, nFev=%d)",
                             bestFitness, nIterations, nFunctionEvals);
    }
}
