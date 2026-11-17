% Example parameter values
beta  = 0.96;
sigma = 2.0;
alpha = 0.36;
delta = 0.08;
amin  = 0;
amax  = 50;
n     = 500;

% Example income process (2-state)
e_vals = [0.5 0.9 1.4 2];            % productivity levels
Pi=[0.7 0.1 0.2 0;
    0.1 0.7 0.1 0.1;
    0.1 0.1 0.7 0.1;
    0 0.2 0.1 0.7;];  % transition matrix (rows sum to 1)

% Call the solver (adjust outputs to your function signature)
[K_star, r_star, w_star, value_function, policy_function, stationary_distribution, mean_assets, it] = solve_ayagari(beta,sigma,alpha,delta,amin,amax,n,e_vals,Pi);

% Display results
fprintf('K* = %.6f, r* = %.6f, w* = %.6f , mean assets = %.6f, iterations = %d\n', K_star, r_star, w_star, mean_assets, it);

%K* = 6.252986, r* = 0.031380, w* = 1.238145 , mean assets = 6.252995, iterations = 48