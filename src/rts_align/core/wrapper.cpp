#include <core/builder.h>

PYBIND11_MODULE(core, m) {
    m.def("construct_graph", &construct_graph,
          "Construct the graph from triangles of Q and K points",
          py::arg("q_pts"), py::arg("k_pts"),                 //
          py::arg("delta") = 5e-3, py::arg("epsilon") = 0.1,  //
          py::arg("min_ratio") = MIN_RATIO_DEFAULT,           //
          py::arg("max_ratio") = MAX_RATIO_DEFAULT);
}
