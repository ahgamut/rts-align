#include <core/builder.h>

static double MIN_RATIO = MIN_RATIO_DEFAULT;
static double MAX_RATIO = MAX_RATIO_DEFAULT;
/* NOTE: LOOK AT THE PARAM ORDERING IN THE MACROS! */

#define TAXICAB_METRIC(a1, a2, b1, b2, c1, c2) \
    (std::fabs((a1) - (a2)) + std::fabs((b1) - (b2)) + std::fabs((c1) - (c2)))

#define EUCDIST_METRIC(a1, a2, b1, b2, c1, c2) \
    (std::hypot((a1) - (a2), std::hypot((b1) - (b2), (c1) - (c2))))

#define ANGLE_COMPARE(a1, a2, b1, b2, c1, c2) \
    EUCDIST_METRIC((a1), (a2), (b1), (b2), (c1), (c2));

#define SRAT_COMPARE(a1, a2, b1, b2, c1, c2) \
    EUCDIST_METRIC((a1), (a2), (b1), (b2), (c1), (c2));

#define SIDE_RATIO(a1, a2, b1, b2, c1, c2) \
    (((a1) / (a2) + (b1) / (b2) + (c1) / (c2)) / 3);

#define BINARY_CMP(n, other, delta, epsilon)  \
    ((sr_compare##n((other)) <= (epsilon)) && \
     (side_ratio##n((other)) <= MAX_RATIO) && \
     (side_ratio##n((other)) >= MIN_RATIO))

#define NUM_THREADS 12
#define ADJMAT_THREAD_SAFE 1

static double hypot3d(double x, double y, double z) {
    return std::hypot(std::hypot(x, y), z);
}

struct Point {
    double x, y, z;
};

struct Triple {
    u32 i : 10;
    u32 j : 10;
    u32 k : 10;
    u32 valid : 1;
    u32 inited : 1;
    double as, bs, cs; /* sides */
    double at, bt, ct; /* angles */
    /* as is the side opposite to at */

    Triple() {
        as = bs = cs = 0;
        i = j = k = 0;
        at = bt = ct = 0;
        valid = 0;  // this->get_valid();
        inited = 0;
    }

    Triple(const Point &a, const Point &b, const Point &c) {
        this->construct(0, 0, 0, a, b, c);
    }

    void construct(const u32 ii, const u32 jj, const u32 kk, const Point &a,
                   const Point &b, const Point &c) {
        /* CALLER NEEDS TO ENSURE THAT ii, jj, kk are < 1024 */
        this->i = ii;
        this->j = jj;
        this->k = kk;
        this->as = hypot3d(c.x - b.x, c.y - b.y, c.z - b.z);
        this->bs = hypot3d(a.x - c.x, a.y - c.y, a.z - c.z);
        this->cs = hypot3d(b.x - a.x, b.y - a.y, b.z - a.z);
        // this->at = stable_angle(a.x - c.x, a.y - c.y, b.x - a.x, b.y - a.y);
        // this->bt = stable_angle(b.x - a.x, b.y - a.y, c.x - b.x, c.y - b.y);
        // this->ct = stable_angle(c.x - b.x, c.y - b.y, a.x - c.x, a.y - c.y);
        this->valid = this->get_valid();
        this->inited = 1;
    }

    bool get_valid() const {
        return (as > MIN_DIST && bs > MIN_DIST && cs > MIN_DIST);
    };

    void coeff_return0(u32 &ii, u32 &jj, u32 &kk) const {
        ii = this->i;
        jj = this->j;
        kk = this->k;
    }
    void coeff_return1(u32 &ii, u32 &jj, u32 &kk) const {
        ii = this->i;
        jj = this->k;
        kk = this->j;
    }
    void coeff_return2(u32 &ii, u32 &jj, u32 &kk) const {
        ii = this->j;
        jj = this->i;
        kk = this->k;
    }
    void coeff_return3(u32 &ii, u32 &jj, u32 &kk) const {
        ii = this->j;
        jj = this->k;
        kk = this->i;
    }
    void coeff_return4(u32 &ii, u32 &jj, u32 &kk) const {
        ii = this->k;
        jj = this->j;
        kk = this->i;
    }
    void coeff_return5(u32 &ii, u32 &jj, u32 &kk) const {
        ii = this->k;
        jj = this->i;
        kk = this->j;
    }

    /* compare angles for similarity */
    double angle_compare0(const Triple &other) const {
        double x = ANGLE_COMPARE(this->at, other.at, this->bt, other.bt,
                                 this->ct, other.ct);
        return x;
    }
    double angle_compare1(const Triple &other) const {
        double x = ANGLE_COMPARE(this->at, other.at, this->bt, other.ct,
                                 this->ct, other.bt);
        return x;
    }
    double angle_compare2(const Triple &other) const {
        double x = ANGLE_COMPARE(this->at, other.bt, this->bt, other.at,
                                 this->ct, other.ct);
        return x;
    }
    double angle_compare3(const Triple &other) const {
        double x = ANGLE_COMPARE(this->at, other.bt, this->bt, other.ct,
                                 this->ct, other.at);
        return x;
    }
    double angle_compare4(const Triple &other) const {
        double x = ANGLE_COMPARE(this->at, other.ct, this->bt, other.bt,
                                 this->ct, other.at);
        return x;
    }
    double angle_compare5(const Triple &other) const {
        double x = ANGLE_COMPARE(this->at, other.ct, this->bt, other.at,
                                 this->ct, other.bt);
        return x;
    }

    /* compare side ratios for similarity */
    double sr_compare0(const Triple &other) const {
        double r1 = this->as / other.as;
        double r2 = this->bs / other.bs;
        double r3 = this->cs / other.cs;
        double x = SRAT_COMPARE(r1, r2, r2, r3, r3, r1);
        return x;
    }
    double sr_compare1(const Triple &other) const {
        double r1 = this->as / other.as;
        double r2 = this->bs / other.cs;
        double r3 = this->cs / other.bs;
        double x = SRAT_COMPARE(r1, r2, r2, r3, r3, r1);
        return x;
    }
    double sr_compare2(const Triple &other) const {
        double r1 = this->as / other.bs;
        double r2 = this->bs / other.as;
        double r3 = this->cs / other.cs;
        double x = SRAT_COMPARE(r1, r2, r2, r3, r3, r1);
        return x;
    }
    double sr_compare3(const Triple &other) const {
        double r1 = this->as / other.bs;
        double r2 = this->bs / other.cs;
        double r3 = this->cs / other.as;
        double x = SRAT_COMPARE(r1, r2, r2, r3, r3, r1);
        return x;
    }
    double sr_compare4(const Triple &other) const {
        double r1 = this->as / other.cs;
        double r2 = this->bs / other.bs;
        double r3 = this->cs / other.as;
        double x = SRAT_COMPARE(r1, r2, r2, r3, r3, r1);
        return x;
    }
    double sr_compare5(const Triple &other) const {
        double r1 = this->as / other.cs;
        double r2 = this->bs / other.as;
        double r3 = this->cs / other.bs;
        double x = SRAT_COMPARE(r1, r2, r2, r3, r3, r1);
        return x;
    }

    /* returns other / this */
    double side_ratio0(const Triple &other) const {
        return SIDE_RATIO(this->as, other.as, this->bs, other.bs, this->cs,
                          other.cs);
    }
    double side_ratio1(const Triple &other) const {
        return SIDE_RATIO(this->as, other.as, this->bs, other.cs, this->cs,
                          other.bs);
    }
    double side_ratio2(const Triple &other) const {
        return SIDE_RATIO(this->as, other.bs, this->bs, other.as, this->cs,
                          other.cs);
    }
    double side_ratio3(const Triple &other) const {
        return SIDE_RATIO(this->as, other.bs, this->bs, other.cs, this->cs,
                          other.as);
    }
    double side_ratio4(const Triple &other) const {
        return SIDE_RATIO(this->as, other.cs, this->bs, other.bs, this->cs,
                          other.as);
    }
    double side_ratio5(const Triple &other) const {
        return SIDE_RATIO(this->as, other.cs, this->bs, other.as, this->cs,
                          other.bs);
    }

    void compare(const Triple &other, u8 check[8], double delta,
                 double epsilon) const {
        for (int i = 0; i < 8; ++i) check[i] = 0;
        /* if any of these are nonzero it means this and other are
         * similar triangles, and hence the ratio of the sides will
         * be roughly equal (barring floating point shenanigans) */
        check[0] = BINARY_CMP(0, other, delta, epsilon);
        check[1] = BINARY_CMP(1, other, delta, epsilon);
        check[2] = BINARY_CMP(2, other, delta, epsilon);
        check[3] = BINARY_CMP(3, other, delta, epsilon);
        check[4] = BINARY_CMP(4, other, delta, epsilon);
        check[5] = BINARY_CMP(5, other, delta, epsilon);
    };
};

static void invert_combi(int n, int i, Triple *t, Point *p) {
    int x = 0;
    int y = 0;
    int z = 0;
    int ii = i;
    /* (x, y, z) is the ith element in the lexicographic ordering
     * of the elements in choose(n, 3). solve for x, y, z.
     * NOTE: 0 <= i, x, y, z < n */

    /* choose(n, 2) elements will start with 0,
     * choose(n-x, 2) elements with start with x */
    for (x = 0; i >= ((n - x - 1) * (n - x - 2)) / 2; ++x) {
        i -= ((n - x - 1) * (n - x - 2)) / 2;
    }

    /* choose ((n-x)-2, 1) elements will start with x, x+1
     * choose ((n-x)-2-y,1) elements will start with x, x+y+1 */
    for (y = 0; i >= ((n - x) - 2 - y); ++y) {
        i -= ((n - x) - 2 - y);
    }

    y = (x + 1) + y;
    z = (y + 1) + i;
    t[ii].construct(x, y, z, p[x], p[y], p[z]);
}

ndarray<u8> construct_graph_3d(ndarray<double> q_pts, ndarray<double> k_pts,
                               double delta, double epsilon, double min_ratio,
                               double max_ratio) {
    /* set ratios before anything */
    MIN_RATIO = min_ratio;
    MAX_RATIO = max_ratio;

    /* delta is in radians */

    /* declare Point arrays and sizes */
    auto q0 = q_pts.unchecked<2>();
    auto k0 = k_pts.unchecked<2>();

    const u32 qlen = q0.shape(0);
    const u32 klen = k0.shape(0);

    if (qlen > NUM_POINTS || klen > NUM_POINTS ||
        qlen * klen > NUM_POINTS * NUM_POINTS) {
        throw std::runtime_error(
            "too many points, might cause memory issues\n");
    }
    if (qlen < 3 || klen < 3) {
        throw std::runtime_error("too few points, cannot calculate\n");
    }

    Point *q = new Point[qlen];
    Point *k = new Point[klen];
    u32 zz;

    for (zz = 0; zz < qlen; zz++) {
        q[zz].x = q0(zz, 0);
        q[zz].y = q0(zz, 1);
        q[zz].z = q0(zz, 2);
    };

    for (zz = 0; zz < klen; zz++) {
        k[zz].x = k0(zz, 0);
        k[zz].y = k0(zz, 1);
        k[zz].z = k0(zz, 2);
    };

    /* DONE COPYING - NOW I don't want the GIL */
    py::gil_scoped_release let_go;

    /* declare Triple arrays and sizes */
    const u32 M = (qlen * (qlen - 1) * (qlen - 2)) / 6;
    const u32 N = (klen * (klen - 1) * (klen - 2)) / 6;
    u32 valid_M = 0;
    u32 valid_N = 0;
    Triple *qt = new Triple[M];
    Triple *kt = new Triple[N];

    const u32 matsize = (qlen * klen);
    u8 *adjmat = new u8[matsize * matsize];
    for (zz = 0; zz < matsize * matsize; ++zz) {
        adjmat[zz] = 0;
    }

    int n_threads = omp_get_max_threads();

#pragma omp parallel num_threads(n_threads)
    {
        u32 ix, iy;
        u8 check[8] = {0};
        u32 i1, j1, k1;
        u32 i2, j2, k2;

        /* fill the first set of triples */
#pragma omp for
        for (ix = 0; ix < M; ++ix) {
            invert_combi(qlen, ix, qt, q);
        }

#pragma omp for reduction(+ : valid_M)
        for (ix = 0; ix < M; ++ix) {
            valid_M += qt[ix].valid;
        }

        /* fill the second set of triples */
#pragma omp for
        for (iy = 0; iy < N; ++iy) {
            invert_combi(klen, iy, kt, k);
        }

#pragma omp for reduction(+ : valid_N)
        for (iy = 0; iy < N; ++iy) {
            valid_N += kt[iy].valid;
        }

#if ADJMAT_THREAD_SAFE
#define ADD_ADJMAT_EDGE(a1, a2, b1, b2)                                    \
    __atomic_store_n(                                                      \
        adjmat + ((a1) * klen + (a2)) * matsize + ((b1) * klen + (b2)), 1, \
        __ATOMIC_SEQ_CST);
#else
#define ADD_ADJMAT_EDGE(a1, a2, b1, b2) \
    adjmat[((a1) * klen + (a2)) * matsize + ((b1) * klen + (b2))] = 1;
#endif

        /* construct the correspondence graph */
#pragma omp for collapse(2)
        for (ix = 0; ix < M; ix++) {
            for (iy = 0; iy < N; iy++) {
                if (qt[ix].valid && kt[iy].valid) {
                    /* the compare call needs to happen here */
                    /* and then you write into adjmat */
                    qt[ix].coeff_return0(i1, j1, k1);
                    qt[ix].compare(kt[iy], check, delta, epsilon);
                    if (check[0]) {
                        kt[iy].coeff_return0(i2, j2, k2);
                        ADD_ADJMAT_EDGE(i1, i2, j1, j2);
                        ADD_ADJMAT_EDGE(j1, j2, k1, k2);
                        ADD_ADJMAT_EDGE(i1, i2, k1, k2);
                    }
                    if (check[1]) {
                        kt[iy].coeff_return1(i2, j2, k2);
                        ADD_ADJMAT_EDGE(i1, i2, j1, j2);
                        ADD_ADJMAT_EDGE(j1, j2, k1, k2);
                        ADD_ADJMAT_EDGE(i1, i2, k1, k2);
                    }
                    if (check[2]) {
                        kt[iy].coeff_return2(i2, j2, k2);
                        ADD_ADJMAT_EDGE(i1, i2, j1, j2);
                        ADD_ADJMAT_EDGE(j1, j2, k1, k2);
                        ADD_ADJMAT_EDGE(i1, i2, k1, k2);
                    }
                    if (check[3]) {
                        kt[iy].coeff_return3(i2, j2, k2);
                        ADD_ADJMAT_EDGE(i1, i2, j1, j2);
                        ADD_ADJMAT_EDGE(j1, j2, k1, k2);
                        ADD_ADJMAT_EDGE(i1, i2, k1, k2);
                    }
                    if (check[4]) {
                        kt[iy].coeff_return4(i2, j2, k2);
                        ADD_ADJMAT_EDGE(i1, i2, j1, j2);
                        ADD_ADJMAT_EDGE(j1, j2, k1, k2);
                        ADD_ADJMAT_EDGE(i1, i2, k1, k2);
                    }
                    if (check[5]) {
                        kt[iy].coeff_return5(i2, j2, k2);
                        ADD_ADJMAT_EDGE(i1, i2, j1, j2);
                        ADD_ADJMAT_EDGE(j1, j2, k1, k2);
                        ADD_ADJMAT_EDGE(i1, i2, k1, k2);
                    }
                }
            }
        }
    }

    // std::cout << valid_M << " valid triangles out of " << M << " in Q\n";
    // std::cout << valid_N << " valid triangles out of " << N << " in K\n";

    /* now I can probably acquire the GIL again */
    py::gil_scoped_acquire omg;

    auto result = ndarray<u8>({matsize, matsize});
    auto res = result.mutable_unchecked<2>();
    u32 x1, x2;
    /* copy adjacency matrix to the result */
    for (x1 = 0; x1 < matsize; ++x1) {
        for (x2 = 0; x2 < matsize; ++x2) {
            res(x1, x2) = adjmat[x1 * matsize + x2];
        }
    }

    /* cleanup memory allocations */
    delete[] q;
    delete[] k;
    delete[] qt;
    delete[] kt;
    delete[] adjmat;

    /* reset ratios to default */
    MIN_RATIO = MIN_RATIO_DEFAULT;
    MAX_RATIO = MAX_RATIO_DEFAULT;

    /* send the answer back */
    return result;
};
