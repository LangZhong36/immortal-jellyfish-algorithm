#include "ija.h"
#include <bits/stdc++.h>
using namespace std;
#define ll long long int

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int T;
    cin >> T;
    while (T--) {
        Cfg c;
        int fid;
        cin >> fid >> c.d >> c.mxit >> c.lb >> c.ub;

        auto fn = [fid](const double* x, int d) { return bench(fid, x, d); };
        Res r;
        optm(c, fn, r);

        cout << fixed << setprecision(10) << r.bestfit << "\n";
        for (int j = 0; j < c.d; j++)
            cout << r.bestpos[j] << " \n"[j == c.d-1];
        cout << "NFev: " << r.nfev << "\n";
    }
    return 0;
}
