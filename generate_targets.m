% Generate the true moments

beta = 0.96;           % discount factor
sigma = 2;             % risk aversion
alpha = 0.33;          % capital share
delta = 0.08;          % depreciation rate
amin = 0;              % borrowing constraint
amax = 50;             % maximum asset value on the grid
n = 500;               % number of grid asset point
rho = 0.8;             % pesistence of the AR(1) process for the shocks
sigma_epsilon = 0.5;   % standard deviation of the innovation in the AR(1) process

true_parameters = [beta, sigma, alpha, delta, amin, amax, n, rho, sigma_epsilon];

[target_moments,true_equilibrium] = compute_moments(true_parameters, 1);

moment_names = {
    '1. Autocorrelation of consumption';
    '2. Average assets of the bottom 1%';
    '3. Average level of assets';
    '4. Standard deviation of assets';
    '5. Standard deviation of income';
    '6. Correlation between income and consumption';
    '7. Corr(c, y) in percentile 1 of wealth';
    '8. Gini of wealth';
    '9. Gini of income';
    '10. Ratio Gini wealth / Gini income';
    '11. Ratio mean wealth / mean income';
    '12. Gini of consumption';
    '13. Ratio Gini consumption / Gini income';
    '14. AR(1) persistence of log income';
    '15. Residual SD of AR(1) log income';
    '16. Corr(c, y) in percentile 50 of wealth';
    '17. Corr(c, y) in percentile 95 of wealth'
};

% Create a table for moments and write to Excel
T_moments = table(moment_names, target_moments, 'VariableNames', {'Description', 'Value'});
writetable(T_moments, 'target_moments.xlsx');

% Define descriptions for the equilibrium values
eq_names = {
    'Aggregate Capital (K*)'; 
    'Interest Rate (r*)'; 
    'Wage (w*)'
};

% true_equilibrium is a 1x3 row vector, so we transpose it to a 3x1 column using '
T_eq = table(eq_names, true_equilibrium', 'VariableNames', {'Description', 'Value'});
writetable(T_eq, 'true_equilibrium.xlsx');

