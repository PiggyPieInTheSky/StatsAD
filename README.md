# AnomDetect

This is a work-in-progress that implements a set of statistical methods for fault-monitoring / anomly-detection from a body of process-control literatures. The current scope of the project is limited to Principal Component Analysis (PCA) based amomaly detection methods. This may change as I progress along with the project.

The current plan of the project is to implement:
- Static PCA for iid data with no autocorrelation
- Dynamic PCA for stationary time series data with non-zero autocorrelation
- Moving window PCA for non-stationary data with 0 autocorrelation
- Common trend method for cointegrated I(1) data