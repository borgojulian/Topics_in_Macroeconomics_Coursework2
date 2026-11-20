% Example parameter values

rng(1);

beta  = 0.9;
sigma = 2.0;
alpha = 0.33;
delta = 0.08;
amin  = -0.05;
amax  = 50;
n     = 500; % number of grid points for assets

rho = 0.4;                  % persistence of the markov chain
sigma_epsilon = 0.2;        % standard deviation of the error term in the AR(1) process

[Pi,log_e,p,arho,asigma] = markovapprox(rho,sigma_epsilon,3,4);
e_vals = exp(log_e);
e_vals = e_vals / (e_vals * p); % Ensures aggregate labor N = 1

% Example income process (2-state)

% Call the solver (adjust outputs to your function signature)
[K_star, r_star, w_star, value_function, policy_function, stationary_distribution, mean_assets, asset_grid, it] = solve_ayagari(beta,sigma,alpha,delta,amin,amax,n,e_vals,Pi);

% Display results
fprintf('K* = %.6f, r* = %.6f, w* = %.6f , mean assets = %.6f, iterations = %d\n', K_star, r_star, w_star, mean_assets, it);

% K* = 10.076136, r* = -0.009805, w* = 1.436024 , mean assets = 10.076144, iterations = 94