#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

struct Neighbor {
    double distance;
    int label;
};

// Calcula la distancia euclidiana entre dos puntos
inline double euclidean_distance(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return std::sqrt(sum);
}

// KNN clasificación
py::array_t<int> knn_classify(py::array_t<double> X_train, py::array_t<int> y_train,
                               py::array_t<double> X_test, int k) {
    auto buf_X_train = X_train.request();
    auto buf_y_train = y_train.request();
    auto buf_X_test = X_test.request();
    
    double* X_train_ptr = static_cast<double*>(buf_X_train.ptr);
    int* y_train_ptr = static_cast<int*>(buf_y_train.ptr);
    double* X_test_ptr = static_cast<double*>(buf_X_test.ptr);
    
    int train_size = buf_X_train.shape[0];
    int test_size = buf_X_test.shape[0];
    int feature_size = buf_X_train.shape[1];
    
    std::vector<int> predictions(test_size);
    
    for (int i = 0; i < test_size; ++i) {
        std::vector<double> test_point(feature_size);
        for (int f = 0; f < feature_size; ++f) {
            test_point[f] = X_test_ptr[i * feature_size + f];
        }
        
        std::vector<Neighbor> neighbors;
        for (int j = 0; j < train_size; ++j) {
            std::vector<double> train_point(feature_size);
            for (int f = 0; f < feature_size; ++f) {
                train_point[f] = X_train_ptr[j * feature_size + f];
            }
            
            double dist = euclidean_distance(test_point, train_point);
            neighbors.push_back({dist, y_train_ptr[j]});
        }
        
        std::sort(neighbors.begin(), neighbors.end(), [](const Neighbor& a, const Neighbor& b) {
            return a.distance < b.distance;
        });
        
        std::vector<int> class_counts(10, 0);
        for (int n = 0; n < k; ++n) {
            class_counts[neighbors[n].label]++;
        }
        
        predictions[i] = std::distance(class_counts.begin(), 
                                       std::max_element(class_counts.begin(), class_counts.end()));
    }
    
    return py::array_t<int>({test_size}, {sizeof(int)}, predictions.data());
}

PYBIND11_MODULE(knn_module, m) {
    m.def("knn_classify", &knn_classify, "K-Nearest Neighbors classification");
}
