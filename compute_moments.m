function [moments,equilibrium_values] = compute_moments(params, seed)
rng(seed);

beta = params(1);           % discount factor
sigma = params(2);             % risk aversion
alpha = params(3);          % capital share
delta = params(4);          % depreciation rate
amin = params(5);              % borrowing constraint
amax = params(6);             % maximum asset value on the grid
n = params(7);               % number of grid asset point
rho = params(8);             % pesistence of the AR(1) process for the shocks
sigma_epsilon = params(9);   % standard deviation of the innovation in the AR(1) process

[Pi,log_e,p,arho,asigma] = markovapprox(rho,sigma_epsilon,3,4);
e_vals = exp(log_e);
e_vals = e_vals / (e_vals * p); % Ensures aggregate labor N = 1

[k_star, r_star, w_star, value_function, policy_function, stationary_distribution, mean_assets, asset_grid, it] = solve_ayagari(beta,sigma,alpha,delta,amin,amax,n,e_vals,Pi);

[cons_path, asset_path, income_path, shock_path] = simulate_model(e_vals,Pi,asset_grid,stationary_distribution,policy_function,w_star,r_star);
    % Flatten panels for cross-sectional distribution moments
    c = cons_path(:);
    a = asset_path(:);
    y = income_path(:);
    
    % Initialize moment vector
    M = zeros(15, 1);
    
    % 1) Autocorrelation of consumption
    c_today = cons_path(:, 1:end-1);
    c_next  = cons_path(:, 2:end);
    M(1) = corr(c_today(:), c_next(:));
    
    % 2) Average assets of the bottom 1% (identifies a_min)
    p1_cutoff = prctile(a, 1);
    M(2) = mean(a(a <= p1_cutoff));
    
    % 3) Average level of assets
    M(3) = mean(a);
    
    % 4) Standard deviation of assets
    M(4) = std(a);
    
    % 5) Standard deviation of income
    M(5) = std(y);
    
    % 6) Correlation between income and consumption
    M(6) = corr(c, y);
    
    % 7) Corr(c, y) in percentiles 1, 50, and 95 of wealth
    % Define localized bins around the percentiles to get enough observations
    idx_p1  = (a <= prctile(a, 1));
    idx_p50 = (a >= prctile(a, 45) & a <= prctile(a, 55));
    idx_p95 = (a >= prctile(a, 95));
    
    M_7_p1  = corr(c(idx_p1), y(idx_p1));
    M_7_p50 = corr(c(idx_p50), y(idx_p50));
    M_7_p95 = corr(c(idx_p95), y(idx_p95));
    
    % We will store the average of the three as moment 7, or you can return them 
    % separately if you want a 17-element moment vector. Here we keep them separate.
    M(7) = M_7_p1;
    M(16) = M_7_p50; % Appended as extra
    M(17) = M_7_p95; % Appended as extra
    
    % 8) Gini of wealth
    M(8) = calculate_gini(a);
    
    % 9) Gini of income
    M(9) = calculate_gini(y);
    
    % 10) Ratio Gini wealth / Gini income
    M(10) = M(8) / M(9);
    
    % 11) Ratio mean wealth / mean income
    M(11) = M(3) / mean(y);
    
    % 12) Gini of consumption
    M(12) = calculate_gini(c);
    
    % 13) Ratio Gini consumption / Gini income
    M(13) = M(12) / M(9);
    
    % Convert income to logs for AR(1) parameters (since shock was ln(e_t))
    log_y_today = log(income_path(:, 1:end-1));
    log_y_next  = log(income_path(:, 2:end));
    
    % 14) AR(1) persistence of log income (estimates rho)
    M(14) = corr(log_y_today(:), log_y_next(:));
    
    % 15) Residual SD of AR(1) log income (estimates sigma_epsilon)
    % Fit a simple linear regression: log_y_next = beta0 + beta1*log_y_today
    p = polyfit(log_y_today(:), log_y_next(:), 1);
    residuals = log_y_next(:) - (p(1) * log_y_today(:) + p(2));
    M(15) = std(residuals);
    
    % Final output vector (17 elements due to splitting Moment 7)
    moments = M;
    equilibrium_values = [k_star, r_star, w_star];
end

function g = calculate_gini(x)
    % Fast, vectorized Gini coefficient calculator for large samples
    % Discard negative values if present (Gini requires strictly non-negative data)
    x = x(x >= 0); 
    if isempty(x)
        g = NaN;
        return;
    end
    
    x = sort(x);
    n = length(x);
    
    % Gini formula based on the Lorenz curve
    cumulative_x = cumsum(x);
    lorenz = cumulative_x / cumulative_x(end);
    
    % Area under the Lorenz curve (trapezoidal rule)
    B = sum(lorenz(1:end-1) + lorenz(2:end)) / (2 * n); 
    
    % Gini is 1 - 2*B
    g = 1 - 2 * B;
end