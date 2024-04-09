
# K Means Clustering Algorithms - Implementation and Comparison

## Objective

The purpose of this project is to implement and compare three different versions of the K Means clustering algorithm: Pure Python, Numpy Arrays (with NumExpr), and Cython. This comparison will focus on analyzing the performance of each implementation, considering factors like execution time and memory usage, especially as the size of the dataset changes.

## Implementation Details

### Versions of K Means Clustering Algorithm

- **Pure Python:** This version utilizes only the standard library functions of Python. It serves as a baseline for performance comparison.

- **Numpy and NumExpr:** This version optimizes calculations using Numpy arrays for efficient data handling and NumExpr for faster computation through optimized expression evaluation.

- **Cython:** This version aims to accelerate the algorithm by leveraging Cython to compile Python code into C, offering significant speed improvements.

### Experimental Setup

- **Number of Clusters (K):** To be defined based on the dataset and specific experiment needs.
- **Dimensionality of Data Points:** The number of features each data point has, to be defined per experiment.
- **Size of the Dataset:** This will vary to explore the scalability and performance impact on the algorithm implementations.

**Hardware and Software Environment:** Details of the computing environment should be specified, including CPU specifications, RAM, operating system, and Python version, to ensure reproducibility of the experiments.

## Profiling

Each implementation will be profiled to measure performance metrics such as execution time and memory usage. The goal is to identify bottlenecks and efficiency differences across implementations.

## Execution and Analysis

- **Execution:** The implementations will be executed with varying dataset sizes, dimensionality, and numbers of clusters to gather comprehensive performance data.

- **Analysis:** Performance data will be compared to understand the scalability and efficiency of each implementation. This analysis will help in identifying the most suitable version of the algorithm for different problem sizes and requirements.

## Documentation

Throughout the implementation process, detailed documentation will be maintained. This documentation will include:

- **Implementation Process:** Step-by-step details of how each version of the algorithm was implemented, including code snippets and explanations of the logic behind each step.

- **Challenges and Solutions:** Any difficulties encountered during the implementation and the strategies used to overcome them.

- **Profiling Results:** Detailed results of the profiling, including graphs and tables that illustrate the performance comparisons.

- **Analysis and Conclusions:** Insights gained from the experiment, highlighting the differences in performance and recommendations on when to use each implementation.

## Requirements

To run the implementations, ensure that you have Python installed on your system along with the following packages:

- Numpy
- NumExpr
- Cython

You can install these packages using the following command:

```
pip install numpy numexpr cython
```

**Note:** For the Cython implementation, additional steps are required to compile the Cython code before execution. Refer to the Cython documentation for guidance on compiling Cython code.

## Usage

Detailed instructions on how to execute each version of the K Means clustering algorithm will be provided, including any necessary commands and parameters.

## Contributing

Contributions to improve the implementations or extend the comparison are welcome. Please follow the project's contribution guidelines for submitting issues or pull requests.

## License

Specify the license under which this project is released, allowing others to understand how they can use, modify, and distribute the code.

---
This README file serves as a comprehensive guide for understanding, executing, and contributing to the project on implementing and comparing different versions of the K Means clustering algorithm.
