#include "ija.h"
#include <iostream>
#include <iomanip>

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int T;
    std::cin >> T;
    while (T--) {
        Cfg c;
        int fid;
        std::cin >> fid >> c.d >> c.mxit >> c.lb >> c.ub;

        auto fn = [fid](const double* x, int d) {
            return bench(fid, x, d);
            };

        Res r;
        optm(c, fn, r);

        std::cout << std::fixed << std::setprecision(10) << r.bestfit << "\n";
        std::cout << r.bestpos[0];
        for (int j = 1; j < c.d; j++) {
            std::cout << " " << r.bestpos[j];
        }
        std::cout << "\n";

        std::cout << "NFev: " << r.nfev << "\n";
    }

    return 0;
}