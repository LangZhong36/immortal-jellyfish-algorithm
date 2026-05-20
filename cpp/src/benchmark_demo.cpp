#include "ija.h"
#include <bits/stdc++.h>
using namespace std;

static const char* FNAME[8] = {
    "Sphere","Rosenbrock","Rastrigin","Ackley","Griewank","Levy","Schwefel","Zakharov"
};

int main() {
    ios::sync_with_stdio(false);
    cout << fixed << setprecision(6);

    printf("%-14s  %14s  %10s  %10s\n", "Function", "Best Fitness", "NFev", "Time(ms)");
    printf("%s\n", string(55, '-').c_str());

    for (int fid = 1; fid <= 8; fid++) {
        Cfg c;
        c.n = 50; c.d = 30; c.mxit = 500;
        c.lb = -100; c.ub = 100;
        if (fid == 3) { c.lb = -5.12; c.ub = 5.12; }
        if (fid == 7) { c.lb = -500;  c.ub = 500;  }
        if (fid == 8) { c.lb = -5;    c.ub = 10;   }

        auto fn = [fid](const double* x, int d) { return bench(fid, x, d); };

        auto t0 = chrono::high_resolution_clock::now();
        Res r;
        optm(c, fn, r);
        auto t1 = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double, milli>(t1-t0).count();

        printf("%-14s  %14.6e  %10lld  %10.1f\n",
               FNAME[fid-1], r.bestfit, r.nfev, ms);
    }
    return 0;
}
