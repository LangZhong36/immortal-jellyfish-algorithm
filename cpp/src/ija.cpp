#include "ija.h"
#include <bits/stdc++.h>
using namespace std;
#define ll long long int

Jly   pop[MAXN];
double arch[MAXAR][MAXD];
double archfit[MAXAR];
int    narchr = 0;
double ltpos[10][MAXD];
double phi[MAXN];
Res    gres;

static mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());

static double randu()       { return uniform_real_distribution<double>(0,1)(rng); }
static double randn()       { return normal_distribution<double>(0,1)(rng); }
static double randrng(double a, double b) { return uniform_real_distribution<double>(a,b)(rng); }

static bool cmpf(const FIdx& a, const FIdx& b) { return a.val < b.val; }

double bench(int fid, const double* x, int d) {
    double s = 0;
    if (fid == 1) {
        for (int i = 0; i < d; i++) s += x[i]*x[i];
    } else if (fid == 2) {
        for (int i = 0; i < d-1; i++)
            s += 100.0*(x[i+1]-x[i]*x[i])*(x[i+1]-x[i]*x[i]) + (1.0-x[i])*(1.0-x[i]);
    } else if (fid == 3) {
        s = 10.0*d;
        for (int i = 0; i < d; i++) s += x[i]*x[i] - 10.0*cos(2.0*M_PI*x[i]);
    } else if (fid == 4) {
        double ss = 0, sc = 0;
        for (int i = 0; i < d; i++) { ss += x[i]*x[i]; sc += cos(2.0*M_PI*x[i]); }
        s = -20.0*exp(-0.2*sqrt(ss/d)) - exp(sc/d) + 20.0 + M_E;
    } else if (fid == 5) {
        double prod = 1.0; double ss = 0;
        for (int i = 0; i < d; i++) { ss += x[i]*x[i]; prod *= cos(x[i]/sqrt((double)(i+1))); }
        s = ss/4000.0 - prod + 1.0;
    } else if (fid == 6) {
        double w0 = 1.0+(x[0]-1.0)/4.0;
        s = sin(M_PI*w0)*sin(M_PI*w0);
        for (int i = 0; i < d-1; i++) {
            double wi = 1.0+(x[i]-1.0)/4.0;
            s += (wi-1.0)*(wi-1.0)*(1.0+10.0*sin(M_PI*wi+1.0)*sin(M_PI*wi+1.0));
        }
        double wn = 1.0+(x[d-1]-1.0)/4.0;
        s += (wn-1.0)*(wn-1.0)*(1.0+sin(2.0*M_PI*wn)*sin(2.0*M_PI*wn));
    } else if (fid == 7) {
        s = 418.9829*d;
        for (int i = 0; i < d; i++) s -= x[i]*sin(sqrt(fabs(x[i])));
    } else {
        double s1 = 0, s2 = 0;
        for (int i = 0; i < d; i++) { s1 += x[i]*x[i]; s2 += 0.5*(i+1)*x[i]; }
        s = s1 + s2*s2 + s2*s2*s2*s2;
    }
    return s;
}

double levy(double beta, int d, double* out) {
    double num = tgamma(1.0+beta)*sin(M_PI*beta/2.0);
    double den = tgamma((1.0+beta)/2.0)*beta*pow(2.0,(beta-1.0)/2.0);
    double sig = pow(num/den, 1.0/beta);
    for (int i = 0; i < d; i++) {
        double u = randn()*sig;
        double v = randn();
        out[i] = u / pow(fabs(v), 1.0/beta);
    }
    return 0;
}

void clmp(double* x, double lb, double ub, int d) {
    for (int i = 0; i < d; i++) {
        if (x[i] < lb) x[i] = lb;
        if (x[i] > ub) x[i] = ub;
    }
}

void init(const Cfg& c, ObjFn fn) {
    narchr = 0;
    for (int i = 0; i < c.n; i++) {
        for (int j = 0; j < c.d; j++)
            pop[i].pos[j] = randrng(c.lb, c.ub);
        pop[i].fit = fn(pop[i].pos, c.d);
        pop[i].age = 0;
        phi[i] = randrng(0, 2.0*M_PI);
    }
}

void attn(double* xi, const double* gs, double inten, double alpha, double lb, double ub, int d) {
    double dist = 0;
    for (int j = 0; j < d; j++) dist += (xi[j]-gs[j])*(xi[j]-gs[j]);
    dist = sqrt(dist);
    for (int j = 0; j < d; j++) {
        double noise = 0.01*randn()*dist;
        xi[j] += alpha*inten*(gs[j]-xi[j]) + noise;
    }
}

void plse(double* xi, double tau, double phas, double lb, double ub, int d) {
    double A = 0.1*(1.0-tau)*(1.0-tau);
    double osc = A*sin(2.0*M_PI*0.5*tau + phas);
    for (int j = 0; j < d; j++)
        xi[j] += osc*(ub-lb);
}

bool trns(double* xi, double& fi, ObjFn fn, double lb, double ub, int d) {
    double opp[MAXD];
    for (int j = 0; j < d; j++) opp[j] = lb+ub-xi[j];
    clmp(opp, lb, ub, d);
    double fo = fn(opp, d);
    if (fo < fi) {
        for (int j = 0; j < d; j++) xi[j] = opp[j];
        fi = fo;
        return true;
    }
    for (int j = 0; j < d; j++) xi[j] = randrng(lb, ub);
    fi = fn(xi, d);
    return false;
}

bool budd(double* xi, double& fi, const double* am, ObjFn fn, double lb, double ub, int d) {
    static double lv[MAXD], off[MAXD];
    levy(1.5, d, lv);
    for (int j = 0; j < d; j++) off[j] = xi[j] + lv[j]*(xi[j]-am[j]);
    clmp(off, lb, ub, d);
    double fo = fn(off, d);
    if (fo < fi) {
        for (int j = 0; j < d; j++) xi[j] = off[j];
        fi = fo;
        return true;
    }
    return false;
}

void strob(int i, const Cfg& c, ObjFn fn, int neph) {
    static double lv[MAXD], cand[MAXD];
    double scale = (c.ub-c.lb)/6.0;
    double bfit = pop[i].fit;
    double bpos[MAXD];
    for (int j = 0; j < c.d; j++) bpos[j] = pop[i].pos[j];
    for (int k = 0; k < neph; k++) {
        levy(c.levy, c.d, lv);
        for (int j = 0; j < c.d; j++) cand[j] = pop[i].pos[j] + randn()*lv[j]*scale;
        clmp(cand, c.lb, c.ub, c.d);
        double fc = fn(cand, c.d);
        if (fc < bfit) { bfit = fc; for (int j = 0; j < c.d; j++) bpos[j] = cand[j]; }
    }
    if (bfit < pop[i].fit) {
        pop[i].fit = bfit;
        for (int j = 0; j < c.d; j++) pop[i].pos[j] = bpos[j];
    }
}

void uarc(const Cfg& c) {
    vector<FIdx> pool;
    for (int i = 0; i < narchr; i++) pool.push_back({archfit[i], i + c.n});
    for (int i = 0; i < c.n; i++) pool.push_back({pop[i].fit, i});
    sort(pool.begin(), pool.end(), cmpf);
    int cap = min((int)pool.size(), c.archsz);
    static double tmp[MAXAR][MAXD];
    static double tmpf[MAXAR];
    for (int k = 0; k < cap; k++) {
        int id = pool[k].idx;
        tmpf[k] = pool[k].val;
        if (id < c.n) for (int j = 0; j < c.d; j++) tmp[k][j] = pop[id].pos[j];
        else          for (int j = 0; j < c.d; j++) tmp[k][j] = arch[id-c.n][j];
    }
    narchr = cap;
    for (int k = 0; k < cap; k++) {
        archfit[k] = tmpf[k];
        for (int j = 0; j < c.d; j++) arch[k][j] = tmp[k][j];
    }
}

void dvrg(const Cfg& c, ObjFn fn) {
    double mean[MAXD] = {};
    for (int i = 0; i < c.n; i++)
        for (int j = 0; j < c.d; j++) mean[j] += pop[i].pos[j];
    for (int j = 0; j < c.d; j++) mean[j] /= c.n;
    double spread = c.ub-c.lb; if (spread <= 0) spread = 1.0;
    double div = 0;
    for (int i = 0; i < c.n; i++)
        for (int j = 0; j < c.d; j++) div += fabs(pop[i].pos[j]-mean[j])/spread;
    div /= (double)(c.n*c.d);
    if (div >= c.dthr) return;
    int nrst = max(1, (int)(c.n*0.2));
    vector<FIdx> worst;
    for (int i = 0; i < c.n; i++) worst.push_back({pop[i].fit, i});
    sort(worst.begin(), worst.end(), cmpf);
    for (int k = c.n-nrst; k < c.n; k++) {
        int id = worst[k].idx;
        for (int j = 0; j < c.d; j++) pop[id].pos[j] = randrng(c.lb, c.ub);
        pop[id].fit = fn(pop[id].pos, c.d);
        pop[id].age = 0;
    }
}

static double sig(double tau) { return 1.0/(1.0+exp(-12.0*(tau-0.5))); }
static double aalph(double tau, double amax, double amin) {
    double w = sig(tau); return amax*(1.0-w)+amin*w;
}
static double lintens(double dsq, double tau) {
    return (1.0/(1.0+0.01*dsq))*exp(-0.5*tau);
}
static int gphase(double tau) {
    if (tau < 0.15) return 0;
    if (tau < 0.50) return 1;
    if (tau < 0.85) return 2;
    return 3;
}
static int gneph(double tau) {
    double frac = (tau-0.15)/(0.50-0.15);
    if (frac < 0) frac = 0; if (frac > 1) frac = 1;
    return max(2, (int)round(5.0 - frac*3.0));
}

void optm(const Cfg& c, ObjFn fn, Res& r) {
    init(c, fn);
    ll nfev = c.n;
    int bestid = 0;
    for (int i = 1; i < c.n; i++) if (pop[i].fit < pop[bestid].fit) bestid = i;
    r.bestfit = pop[bestid].fit;
    for (int j = 0; j < c.d; j++) r.bestpos[j] = pop[bestid].pos[j];
    r.curve[0] = r.bestfit;
    uarc(c);

    static double lv[MAXD];

    for (int t = 1; t <= c.mxit; t++) {
        double tau = (double)t/c.mxit;
        int phase = gphase(tau);
        int neph  = gneph(tau);
        double alpha = aalph(tau, c.amax, c.amin);

        vector<FIdx> elite;
        for (int i = 0; i < c.n; i++) elite.push_back({pop[i].fit, i});
        sort(elite.begin(), elite.end(), cmpf);
        int nlt = min(c.topk, c.n);
        for (int k = 0; k < nlt; k++)
            for (int j = 0; j < c.d; j++) ltpos[k][j] = pop[elite[k].idx].pos[j];

        for (int i = 0; i < c.n; i++) {
            double* xi = pop[i].pos;
            double& fi = pop[i].fit;
            double prev = fi;

            if (phase == 0) {
                levy(c.levy, c.d, lv);
                double sc = 0.01*(c.ub-c.lb);
                for (int j = 0; j < c.d; j++) xi[j] += sc*lv[j]*(xi[j]-r.bestpos[j]);
                clmp(xi, c.lb, c.ub, c.d);
                fi = fn(xi, c.d); nfev++;

            } else if (phase == 1) {
                strob(i, c, fn, neph);
                nfev += neph;

            } else if (phase == 2) {
                int gs = elite[0].idx;
                double dsq = 0;
                for (int j = 0; j < c.d; j++) dsq += (xi[j]-pop[gs].pos[j])*(xi[j]-pop[gs].pos[j]);
                double inten = lintens(dsq, tau);
                attn(xi, pop[gs].pos, inten, alpha, c.lb, c.ub, c.d);
                plse(xi, tau, phi[i], c.lb, c.ub, c.d);
                clmp(xi, c.lb, c.ub, c.d);
                if (pop[i].age >= c.agemax) {
                    trns(xi, fi, fn, c.lb, c.ub, c.d);
                    pop[i].age = 0; nfev += 2;
                } else {
                    fi = fn(xi, c.d); nfev++;
                }
                if (narchr > 0) {
                    int ar = (int)(randu()*narchr) % narchr;
                    budd(xi, fi, arch[ar], fn, c.lb, c.ub, c.d); nfev++;
                }

            } else {
                double spread = (c.ub-c.lb)*0.05*(1.0-tau)*(1.0-tau);
                double cand[MAXD];
                for (int j = 0; j < c.d; j++) cand[j] = r.bestpos[j] + spread*randn();
                clmp(cand, c.lb, c.ub, c.d);
                double fc = fn(cand, c.d); nfev++;
                if (fc < fi) { for (int j = 0; j < c.d; j++) xi[j] = cand[j]; fi = fc; }
                if (narchr > 0) {
                    int ar = (int)(randu()*narchr) % narchr;
                    budd(xi, fi, arch[ar], fn, c.lb, c.ub, c.d); nfev++;
                }
            }

            if (fi < prev) pop[i].age = 0; else pop[i].age++;
            if (fi < r.bestfit) {
                r.bestfit = fi;
                for (int j = 0; j < c.d; j++) r.bestpos[j] = xi[j];
            }
        }

        dvrg(c, fn);
        uarc(c);
        r.curve[t] = r.bestfit;
    }
    r.niter = c.mxit;
    r.nfev  = nfev;
}
