#ifndef RTS_ALIGN_BUILDER_H
#define RTS_ALIGN_BUILDER_H

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <set>

/* after headers, pybind */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
/* after pybind, numpy */
#include <pybind11/numpy.h>

namespace py = pybind11;

template <typename T>
using ndarray = py::array_t<T, py::array::c_style>;

using u32 = uint32_t;
using u64 = uint64_t;
using u16 = uint16_t;
using u8 = uint8_t;

static constexpr double MIN_RATIO_DEFAULT = 0.5;
static constexpr double MAX_RATIO_DEFAULT = 2.5;
static constexpr double MIN_DIST = 1e-2;
static constexpr double MAX_DIST = 1e8;
static constexpr double MIN_COSINE_DIST = 1e-3;
static constexpr double MAX_COSINE_DIST = 2.0;
static constexpr double PI = 3.14159265358979;
static constexpr u32 NUM_POINTS = 384; /* technically 1024 */

ndarray<u8> construct_graph_2d(ndarray<double> q_pts, ndarray<double> k_pts,
                               double delta, double epsilon, double min_ratio,
                               double max_ratio);
ndarray<u8> construct_graph_3d(ndarray<double> q_pts, ndarray<double> k_pts,
                               double delta, double epsilon, double min_ratio,
                               double max_ratio);
ndarray<u8> construct_graph(ndarray<double> q_pts0, ndarray<double> k_pts0,
                            ndarray<double> q_dist0, ndarray<double> k_dist0,
                            double epsilon, bool distancesAreCosine);

#endif
