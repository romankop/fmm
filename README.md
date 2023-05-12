# fmm
The library containing generators of financial market models.

Currently the library covers the classic financial market models.
Multidimensional generation of the following models is available:

0. Binary Tree Model <br /> $S_t = S_0 \cdot u ^{N_u} d^{N_d}$
1. Black-Scholes Model (1973) <br /> $dS_t = \mu S_t dt + \sigma S_t dW_t$
2. Ornstein-Uhlenbeck Model (1930) <br /> $dS_t = a(b - S_t) dt + \sigma dW_t$ 
3. Cox–Ingersoll–Ross Model (1985) <br /> $dS_t = a (b -S_t) dt + \sigma \sqrt{S_t} dW_t$ 
4. Courtadon Model (1982) <br /> $dS_t = a (b -S_t) dt + \sigma S_t dW_t$
5. Ho-Lee Model (1986) <br /> $dS_t = \mu dt + \sigma dW_t$ 
6. Constant Elasticity of Variance Model (1975) <br /> $dS_t = \mu S_t dt + \sigma {S_t}^{\alpha} dW_t$
7. Cox–Ingersoll–Ross Model (1980) <br /> $dS_t = \sigma {S_t}^{\frac{3}{2}} dW_t$
8. Cox–Ingersoll–Ross Model (1980) Extension with Constant Elasticity of Variance <br /> $dS_t = \sigma {S_t}^{\alpha} dW_t$
9. Dothan Model (1978, or Black Futures Model) <br /> $dS_t = \sigma S_t dW_t$
10. Chan–Karolyi–Longstaff–Sanders Model (1992) <br /> $dS_t = a (b - S_t) dt + \sigma {S_t}^{\alpha} dW_t$
11. Marsh-Rosenfeld Model (1983) <br /> $dS_t = (b S_t + a {S_t}^{\alpha - 1}) dt + \sigma {S_t}^{\frac{\alpha}{2}} dW_t$
12. Duffie and Kan Single Factor Model (1996) <br /> $dS_t = a (b - S_t) dt + \sqrt{\sigma + \gamma S_t} dW_t$
13. Constantinides Model (1992) <br /> $dS_t = [a (b - S_t) + \gamma S_t^2] dt + (\sigma + \gamma S_t) dW_t$
14. Black–Karasinski Model (1991) <br /> $dS_t = a (b - \log{S_t}) dt + \sigma \sqrt{S_t} dW_t$
15. Brennan-Schwartz Model (1979) <br /> $dS_t = a S_t (b - \log{S_t}) dt + \sigma S_t dW_t$
16. Heston Model (1993) <br /> $dS_t = \mu S_t dt + \sqrt{V_t} S_t dW_t$  <br /> $dV_t = a (b - V_t) dt + \sigma \sqrt{V_t} dZ_t$ 
17. Chen Model (1996) <br /> $dS_t = k (\theta_t - S_t) dt + \sqrt{V_t} \sqrt{S_t} dW^1_t$  <br /> $d\theta_t = a_1 (b_1 - \theta_t) dt + \sigma_1 \sqrt{\theta_t} dW^2_t$  <br /> $dV_t = a_2 (b_2 - V_t) dt + \sigma_2 \sqrt{V_t} dW^3_t$ 
18. SABR ("stochastic alpha, beta, rho") Model (2002)  <br /> $dS_t = V_t {S_t}^{\alpha} dW_t$  <br /> $dV_t = \sigma V_t dZ_t$

