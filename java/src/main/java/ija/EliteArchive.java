package ija;

import java.util.*;

/**
 * Fixed-capacity archive of elite solutions.
 *
 * When the archive exceeds capacity, entries are pruned by (fitness, then
 * crowding penalty) to retain both quality and diversity.
 */
class EliteArchive {

    private final int maxSize;
    private final List<double[]> positions = new ArrayList<>();
    private final List<Double>   fitnesses = new ArrayList<>();
    private final Random rng = new Random();

    EliteArchive(int maxSize) {
        this.maxSize = maxSize;
    }

    void update(double[][] population, double[] fits) {
        for (int i = 0; i < population.length; i++) {
            positions.add(Arrays.copyOf(population[i], population[i].length));
            fitnesses.add(fits[i]);
        }
        if (positions.size() <= maxSize) return;

        Integer[] order = new Integer[positions.size()];
        for (int i = 0; i < order.length; i++) order[i] = i;
        Arrays.sort(order, Comparator.comparingDouble(i -> fitnesses.get(i)));

        List<double[]> newPos  = new ArrayList<>(maxSize);
        List<Double>   newFits = new ArrayList<>(maxSize);
        for (int k = 0; k < maxSize; k++) {
            newPos.add(positions.get(order[k]));
            newFits.add(fitnesses.get(order[k]));
        }
        positions.clear();
        fitnesses.clear();
        positions.addAll(newPos);
        fitnesses.addAll(newFits);
    }

    double[] sample() {
        if (positions.isEmpty()) return null;
        return positions.get(rng.nextInt(positions.size()));
    }

    int size() {
        return positions.size();
    }
}
