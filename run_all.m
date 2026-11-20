
seed = 1;

% Generate the true parameters
run generate_targets.m   % this code solve the model for the true parameters 
                         % and calculate the target moments.

%Estimate the parameters by Simulated Method of Moments, starting from the
%initial guess for 
fixed_params = struct('alpha', 0.33, 'delta', 0.08, 'sigma', 2.0, 'amax', 50, 'n', 500);
initial_guess = [0.4, 0.2, 0.9, -0.05]; % [rho, sigma_epsilon, beta, amin]

[theta_hat, fval, exitflag, output, diag_info] = smm_estimation_parameters(target_moments, fixed_params, initial_guess, seed);
