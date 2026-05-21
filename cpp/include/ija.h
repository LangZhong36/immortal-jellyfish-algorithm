#pragma once

#include <iostream>
#include <vector>

static const int MAXN = 505;
static const int MAXD = 305;
static const int MAXAR = 55;

struct Jly {
    double pos[MAXD];
    double fit;
    int age;
};

struct FIdx {
    double val;
    int idx;
    bool operator<(const FIdx& o) const {
        return val < o.val;
    }
};

struct Cfg {
    int n, d, mxit;
    double lb, ub;
    int topk, archsz, agemax, neph;
    double levy, lmbd, amax, amin, dthr;
    Cfg() : n(50), d(30), mxit(500),
        lb(-100), ub(100),
        topk(3), archsz(15), agemax(25), neph(3),
        levy(1.5), lmbd(0.5), amax(2.0), amin(0.05), dthr(0.005) {
    }
};

struct Res {
    double bestfit;
    double bestpos[MAXD];
    double curve[5005];
    int niter;
    long long int nfev;
};

extern Jly pop[MAXN];
extern double arch[MAXAR][MAXD];
extern double archfit[MAXAR];
extern int narchr;
extern double ltpos[10][MAXD];
extern double phi[MAXN];
extern Res gres;

typedef double (*ObjFn)(const double*, int);

double bench(int fid, const double* x, int d);

double levy(double beta, int d, double* out);
void   clmp(double* x, double lb, double ub, int d);
void   init(const Cfg& c, ObjFn fn);
void   updt(const Cfg& c, ObjFn fn, int t);
void   dvrg(const Cfg& c, ObjFn fn);
void   attn(double* xi, const double* gs, double inten, double alpha, double lb, double ub, int d);
void   plse(double* xi, double tau, double phas, double lb, double ub, int d);
bool   trns(double* xi, double& fi, ObjFn fn, double lb, double ub, int d);
bool   budd(double* xi, double& fi, const double* am, ObjFn fn, double lb, double ub, int d);
void   strob(int i, const Cfg& c, ObjFn fn, int neph);
void   uarc(const Cfg& c);
void   optm(const Cfg& c, ObjFn fn, Res& r);
