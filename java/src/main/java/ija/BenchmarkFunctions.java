package ija;

/**
 * Standard benchmark functions for continuous optimization.
 *
 * All functions accept a double[] and return a double.
 * Global minimum is at or near 0 for all functions.
 */
public class BenchmarkFunctions {

    public static double sphere(double[] x) {
        double s = 0;
        for (double v : x) s += v * v;
        return s;
    }

    public static double rosenbrock(double[] x) {
        double s = 0;
        for (int i = 0; i < x.length - 1; i++) {
            double t1 = x[i + 1] - x[i] * x[i];
            double t2 = 1.0 - x[i];
            s += 100.0 * t1 * t1 + t2 * t2;
        }
        return s;
    }

    public static double rastrigin(double[] x) {
        double s = 10.0 * x.length;
        for (double v : x) s += v * v - 10.0 * Math.cos(2.0 * Math.PI * v);
        return s;
    }

    public static double ackley(double[] x) {
        int d = x.length;
        double sumSq = 0, sumCos = 0;
        for (double v : x) { sumSq += v * v; sumCos += Math.cos(2.0 * Math.PI * v); }
        return -20.0 * Math.exp(-0.2 * Math.sqrt(sumSq / d))
               - Math.exp(sumCos / d) + 20.0 + Math.E;
    }

    public static double griewank(double[] x) {
        double sumSq = 0, prod = 1.0;
        for (int i = 0; i < x.length; i++) {
            sumSq += x[i] * x[i];
            prod  *= Math.cos(x[i] / Math.sqrt(i + 1));
        }
        return sumSq / 4000.0 - prod + 1.0;
    }

    public static double levy(double[] x) {
        int d = x.length;
        double w0 = 1.0 + (x[0] - 1.0) / 4.0;
        double s   = Math.pow(Math.sin(Math.PI * w0), 2);
        for (int i = 0; i < d - 1; i++) {
            double wi = 1.0 + (x[i] - 1.0) / 4.0;
            s += (wi - 1.0) * (wi - 1.0) * (1.0 + 10.0 * Math.pow(Math.sin(Math.PI * wi + 1.0), 2));
        }
        double wn = 1.0 + (x[d - 1] - 1.0) / 4.0;
        s += (wn - 1.0) * (wn - 1.0) * (1.0 + Math.pow(Math.sin(2.0 * Math.PI * wn), 2));
        return s;
    }

    public static double schwefel(double[] x) {
        double s = 418.9829 * x.length;
        for (double v : x) s -= v * Math.sin(Math.sqrt(Math.abs(v)));
        return s;
    }

    public static double zakharov(double[] x) {
        double s1 = 0, s2 = 0;
        for (int i = 0; i < x.length; i++) {
            s1 += x[i] * x[i];
            s2 += 0.5 * (i + 1) * x[i];
        }
        return s1 + s2 * s2 + s2 * s2 * s2 * s2;
    }

    public static final String[] NAMES = {
        "Sphere", "Rosenbrock", "Rastrigin", "Ackley",
        "Griewank", "Levy", "Schwefel", "Zakharov"
    };

    public static double evaluate(int fid, double[] x) {
        switch (fid) {
            case 0: return sphere(x);
            case 1: return rosenbrock(x);
            case 2: return rastrigin(x);
            case 3: return ackley(x);
            case 4: return griewank(x);
            case 5: return levy(x);
            case 6: return schwefel(x);
            case 7: return zakharov(x);
            default: throw new IllegalArgumentException("Unknown function id: " + fid);
        }
    }
}
