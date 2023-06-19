Principal Component Analysis and Cointegration
==============================================

This article outlines the relation between Principal Component Anlaysis (PCA) and cointegration based on Chigira (2008), and the basic idea behind Chigira cointegration test. 

Recall that a prerequisite of cointegrated series is that every series has to be integrated of the same order. 
Let's denote :math:`I(d)` for integration of order :math:`d` (`see here <https://en.wikipedia.org/wiki/Order_of_integration>`_). 
This means after :math:`d` times of taking difference (i.e. :math:`y_{t} - y_{t-1}`), the series becomes (*weak*) `stationary <https://en.wikipedia.org/wiki/Stationary_process#Weak_or_wide-sense_stationarity>`_. 
Here we will only explore the case of I(1) series. In this case, cointegration simply says that there exists a linear combination of these integrated series such that the combination is stationary.

Further recall that (*linear*) `Principal Component Analysis <https://en.wikipedia.org/wiki/Principal_component_analysis>`_ seeks to find a `unit vector <https://en.wikipedia.org/wiki/Unit_vector>`_ that maximizes the variance after the vector is applied as weights to the original series. 
What immediately follows are:
- The maximized variance is the eigenvalue from the eigen-decomposition of the variance-covariance matrix of the original data. 

- The eigenvectors are the weights. These vectors may also be known as *factor loadings* in other fields of studies.

- A principal component is the linear combination of the original series with eigenvectors as the weights.

- The 2nd (or 3rd, 4th, ...) largest variance is the eignvalue of the variance-covariance matrix based on the residual data that the previous principal component(s) cannot explain. 

- Because these eigenvectors are `orthonormal <https://en.wikipedia.org/wiki/Orthonormality>`_ (i.e. the dot product is 0 between any two and 1 with  itself), the principal components are orthogonal as the result.



Chigira (2008) proved that if the last few principal components are stationary, the original data are in fact cointegrated, provided each series is I(1). 
Furthermore, the number of stationary principal components equals the number of cointegration vectors.

Let :math:`\pmb{x}_{t}=[x^1_t, x^2_t, ..., x^l_t ]^\prime` denote a vector of variables from :math:`x^1` to :math:`x^l` for time period :math:`t`, properly demeaned and detrended. 
Just as Chigira (2008), we use a VAR model with a constant, a time trend and no lagged terms to model out the mean and trend:

.. math::
    \begin{equation*}
        \pmb{y}_t = \mu + \sigma t + \pmb{x}_{t}
    \end{equation*}

where :math:`\pmb{x}_{t}` is simply the residual of VAR on the original series :math:`\pmb{y}_t`.

Let :math:`\pmb{B} = [ \pmb{v}_1, ... , \pmb{v}_m]` be an :math:`m` by :math:`l` matrix of PCA eigenvectors, where :math:`m` is the intended number of principal components to keep. Thus :math:`m \le l`. 
The columns of :math:`\pmb{B}` are ordered descendingly according to the eigenvalues. 
The PCA components are defined as:

.. math::
    \begin{align*}
        \pmb{V}_t &= \pmb{B}^\prime \pmb{x}_{t}\\
        &= \begin{bmatrix}
            | & & |\\
            \pmb{v}_1^\prime \pmb{x}_{t}  &... & \pmb{v}_m^\prime \pmb{x}_{t}\\
            | & & |\\
        \end{bmatrix}
    \end{align*}


Chigira cointegration test then becomes a test for stationary principal components. This implementation of the Chigira test utilizes the following precedure:

1. :math:`i=0`, :math:`r=0`

2. Run the `Dickey-Fuller <https://en.wikipedia.org/wiki/Dickey%E2%80%93Fuller_test>`_ unit root test on the principal component with :math:`i`-th *smallest* eigenvalue

3. If the test shows unit root, return :math:`r`

4. If the test does not show a unit root, let :math:`r = i + 1`

5. Unless :math:`i` = :math:`m`, increase :math:`i` by 1

The return variable :math:`r` represents the number of cointegrated vectors as in `Johansen cointegration <https://en.wikipedia.org/wiki/Johansen_test>`_ test.


As Lin et al. (2019) noted, the benifits of the Chigira cointegration test includes:

- the ability to test a large number of variables even when there is a small number of time periods in data,

- the structural flexibility compared to Johansen cointegration test since the latter assumes an autoregressive structure of the data.

Below is an example code to test cointegration of a series of data.