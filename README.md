# RFF
Random foiurier features in RKHS

This code is a test of concept for the use of random fourier features to speed up kernel machines in Reproducing Kernel Hilbert Space.  By Bochner's theorem, andy shift invariant kernel function k(x,y) can be approximated by <Z(x,t), Z(y,t)> where Z(x,t) is a linear combination of fourier basis functions.  This implementation does not randomly sample vectors as in MC simulations but finds optimal realizations of the random variables by minimizing an SVM objective function.
