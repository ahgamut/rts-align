#include <core/builder.h>

#define TAXICAB_METRIC(a1, a2, b1, b2, c1, c2) \
    (std::fabs((a1) - (a2)) + std::fabs((b1) - (b2)) + std::fabs((c1) - (c2)))

#define EUCDIST_METRIC(a1, a2, b1, b2, c1, c2) \
    (std::hypot((a1) - (a2), std::hypot((b1) - (b2), (c1) - (c2))))

#define BINARY_CMP(n, other, epsilon) (part_compare##n((other)) <= (epsilon))

#define CHECK_CLAMPED(x, low, hi) (((x) >= (low)) && ((x) <= (hi)))
#define CHECK_CLAMPED_COS(x) \
    CHECK_CLAMPED((x), MIN_COSINE_DIST, MAX_COSINE_DIST)
#define CHECK_CLAMPED_RAW(x) CHECK_CLAMPED((x), MIN_DIST, MAX_DIST)

#define NUM_THREADS 12
#define ADJMAT_THREAD_SAFE 1

class RawNDArray {
   private:
    double *ptr;
    u32 nrows;
    u32 ncols;
    u32 row_stride;
    u32 col_stride;
    u32 row_stride_elements;

    RawNDArray() {
        ptr = nullptr;
        nrows = 0;
        ncols = 0;
        row_stride = 0;
        col_stride = 0;
    }

   public:
    static RawNDArray fromPyArray(ndarray<double> &arr) {
        RawNDArray raw;
        pybind11::buffer_info buf = arr.request();
        //
        raw.ptr = static_cast<double *>(buf.ptr);
        raw.nrows = buf.shape[0];
        raw.ncols = buf.shape[1];
        raw.row_stride = buf.strides[0];
        raw.col_stride = buf.strides[1];
        raw.row_stride_elements = raw.row_stride / sizeof(double);
        return raw;
    }

    double operator()(u32 i, u32 j) const {
        return *(ptr + i * row_stride_elements + j);
    }

    u32 get_nrows() const { return nrows; }
    u32 get_ncols() const { return ncols; }
};

static std::string arrayCheckErrorReasons[] = {
    " is not a 2-D array!",           //
    " has too many points!",          //
    " has too few points!",           //
    " incorrect number of columns!",  //
};

RawNDArray getRawAfterCheck(ndarray<double> &arr, size_t ncol) {
    if (arr.ndim() != 2) {
        throw std::runtime_error(arrayCheckErrorReasons[0]);
    }
    auto a0 = arr.unchecked<2>();
    if (a0.shape(0) > NUM_POINTS) {
        throw std::runtime_error(arrayCheckErrorReasons[1]);
    }
    if (a0.shape(0) < 3) {
        throw std::runtime_error(arrayCheckErrorReasons[2]);
    }
    if (ncol != a0.shape(1)) {
        throw std::runtime_error(arrayCheckErrorReasons[3]);
    }
    return RawNDArray::fromPyArray(arr);
}

struct Triple {
    u32 i : 10;
    u32 j : 10;
    u32 k : 10;
    u32 valid : 1;
    u32 inited : 1;
    double at, bt, ct;

    Triple() {
        i = j = k = 0;
        at = bt = ct = 0;
        valid = 0;  // this->get_valid();
        inited = 0;
    }

    void construct_at(const u32 ii, const u32 jj, const u32 kk, RawNDArray &pts,
                      RawNDArray &dist) {
        /* CALLER NEEDS TO ENSURE THAT ii, jj, kk are < 1024 */
        this->i = ii;
        this->j = jj;
        this->k = kk;
        this->inited = 1;
        //
        this->at = dist(ii, jj);
        this->bt = dist(jj, kk);
        this->ct = dist(kk, ii);
        this->valid = 1;
        this->valid = this->valid && CHECK_CLAMPED_COS(at);
        this->valid = this->valid && CHECK_CLAMPED_COS(bt);
        this->valid = this->valid && CHECK_CLAMPED_COS(ct);
    }

    void construct_st(const u32 ii, const u32 jj, const u32 kk, RawNDArray &pts,
                      RawNDArray &dist) {
        /* CALLER NEEDS TO ENSURE THAT ii, jj, kk are < 1024 */
        this->i = ii;
        this->j = jj;
        this->k = kk;
        this->inited = 1;
        double as = dist(ii, jj);
        double bs = dist(jj, kk);
        double cs = dist(kk, ii);
        this->valid = 1;
        this->valid = this->valid && CHECK_CLAMPED_RAW(as);
        this->valid = this->valid && CHECK_CLAMPED_RAW(bs);
        this->valid = this->valid && CHECK_CLAMPED_RAW(cs);
        if (this->valid) {
            this->at = std::log(as) - std::log(bs);
            this->bt = std::log(bs) - std::log(cs);
            this->ct = std::log(cs) - std::log(as);
        }
    }

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

    double part_compare0(const Triple &other) const {
        double x = EUCDIST_METRIC(this->at, other.at, this->bt, other.bt,
                                  this->ct, other.ct);
        return x;
    }
    double part_compare1(const Triple &other) const {
        double x = EUCDIST_METRIC(this->at, other.at, this->bt, other.ct,
                                  this->ct, other.bt);
        return x;
    }
    double part_compare2(const Triple &other) const {
        double x = EUCDIST_METRIC(this->at, other.bt, this->bt, other.at,
                                  this->ct, other.ct);
        return x;
    }
    double part_compare3(const Triple &other) const {
        double x = EUCDIST_METRIC(this->at, other.bt, this->bt, other.ct,
                                  this->ct, other.at);
        return x;
    }
    double part_compare4(const Triple &other) const {
        double x = EUCDIST_METRIC(this->at, other.ct, this->bt, other.bt,
                                  this->ct, other.at);
        return x;
    }
    double part_compare5(const Triple &other) const {
        double x = EUCDIST_METRIC(this->at, other.ct, this->bt, other.at,
                                  this->ct, other.bt);
        return x;
    }

    void compare(const Triple &other, u8 check[8], double epsilon) const {
        for (int i = 0; i < 8; ++i) check[i] = 0;
        /* if any of the below bytes are nonzero it means
         * this and other have points of similar triangles*/
        check[0] = BINARY_CMP(0, other, epsilon);
        check[1] = BINARY_CMP(1, other, epsilon);
        check[2] = BINARY_CMP(2, other, epsilon);
        check[3] = BINARY_CMP(3, other, epsilon);
        check[4] = BINARY_CMP(4, other, epsilon);
        check[5] = BINARY_CMP(5, other, epsilon);
    };
};

static void invert_combi(int n, int i, Triple *t,  //
                         RawNDArray &pts, RawNDArray &dist,
                         bool distancesAreCosine) {
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
    if (distancesAreCosine) {
        t[ii].construct_at(x, y, z, pts, dist);
    } else {
        t[ii].construct_st(x, y, z, pts, dist);
    }
}

ndarray<u8> construct_graph(ndarray<double> q_pts0, ndarray<double> k_pts0,
                            ndarray<double> q_dist0, ndarray<double> k_dist0,
                            double epsilon, bool distancesAreCosine) {
    /* check array shapes */
    RawNDArray q_pts = getRawAfterCheck(q_pts0, q_pts0.shape(1));
    RawNDArray q_dist = getRawAfterCheck(q_dist0, q_pts.get_nrows());
    RawNDArray k_pts = getRawAfterCheck(k_pts0, q_pts0.shape(1));
    RawNDArray k_dist = getRawAfterCheck(k_dist0, k_pts.get_nrows());

    const u32 qlen = q_pts.get_nrows();
    const u32 klen = k_pts.get_nrows();

    /* declare Triple arrays and sizes */
    const u32 M = (qlen * (qlen - 1) * (qlen - 2)) / 6;
    const u32 N = (klen * (klen - 1) * (klen - 2)) / 6;
    u32 valid_M = 0;
    u32 valid_N = 0;
    Triple *qt = new Triple[M];
    Triple *kt = new Triple[N];

    const u32 matsize = (qlen * klen);
    u32 zz = 0;
    u8 *adjmat = new u8[matsize * matsize];
    for (zz = 0; zz < matsize * matsize; ++zz) {
        adjmat[zz] = 0;
    }

    int n_threads = omp_get_max_threads();
    {
        /* DONE COPYING - NOW I don't want the GIL */
        py::gil_scoped_release let_go;

#pragma omp parallel num_threads(n_threads)
        {
            u32 ix, iy;
            u8 check[8] = {0};
            u32 i1, j1, k1;
            u32 i2, j2, k2;

            /* fill the first set of triples */
#pragma omp for
            for (ix = 0; ix < M; ++ix) {
                invert_combi(qlen, ix, qt, q_pts, q_dist, distancesAreCosine);
            }

#pragma omp for reduction(+ : valid_M)
            for (ix = 0; ix < M; ++ix) {
                valid_M += qt[ix].valid;
            }

            /* fill the second set of triples */
#pragma omp for
            for (iy = 0; iy < N; ++iy) {
                invert_combi(klen, iy, kt, k_pts, k_dist, distancesAreCosine);
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
                        qt[ix].compare(kt[iy], check, epsilon);
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
    }
    // std::cout << valid_M << " valid triangles out of " << M << " in Q\n";
    // std::cout << valid_N << " valid triangles out of " << N << " in K\n";
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
    delete[] qt;
    delete[] kt;
    delete[] adjmat;

    /* send the answer back */
    return result;
};
