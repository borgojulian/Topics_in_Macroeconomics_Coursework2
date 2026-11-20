function [theta_hat, fval, exitflag, output, diag_info] = smm_estimation_parameters(target_moments, fixed_params, theta0, seed)
% Estimates theta = [rho, sigma_epsilon, beta, amin] via SMM
% using the Nelder-Mead algorithm (fminsearch) with W = I.
%
% Inputs:
%   target_moments : (M x 1) vector of target moments m_data
%   fixed_params   : struct with fields:
%                      .alpha, .delta, .sigma, .amax, .n
%   theta0         : (1 x 4) or (4 x 1) initial guess [rho, sigma_epsilon, beta, amin]
%                    Default from coursework: [0.4, 0.2, 0.9, -0.05]
%   seed           : random seed used in every objective evaluation (default = 1)

% Parameter bounds: [rho, sigma_epsilon, beta, amin]
lb = [-0.99,  1e-4,  0.50, -5.0];
ub = [ 0.99,  2.00,  0.995, 2.0];

% Identity weighting matrix W = I (Section 5.1.2)
M = length(target_moments);
W = eye(M);

% Objective function handle with bounds, safeguards, and fixed seed
obj_fun = @(th) smm_objective(th, target_moments, W, fixed_params, lb, ub, seed);

% Nelder-Mead options: MaxIter = 500 and tolerance levels (Section 5.1.4 - 5.1.5)
options = optimset('Display', 'iter', ...
                   'MaxIter', 500, ...
                   'MaxFunEvals', 2000, ...
                   'TolX', 1e-6, ...
                   'TolFun', 1e-6);

% Run Nelder-Mead optimization
[theta_hat, fval, exitflag, output] = fminsearch(obj_fun, theta0, options);

% Package diagnostics
diag_info = struct();
diag_info.theta0 = theta0;
diag_info.lb = lb;
diag_info.ub = ub;
diag_info.seed = seed;
diag_info.W = W;
diag_info.iterations = output.iterations;
diag_info.funcCount = output.funcCount;
diag_info.message = output.message;
end


function Q = smm_objective(theta, m_data, W, fixed, lb, ub, seed)
% SMM_OBJECTIVE Evaluates Q(theta) = [m_data - m_model(theta)]' * W * [m_data - m_model(theta)]
% with economic parameter bound penalties and numerical failure safeguards.

penalty = 1e12;

% 2. Check economic parameter bounds and apply smooth + large quadratic penalty if violated
below_lb = max(0, lb - theta);
above_ub = max(0, theta - ub);
viol = sum(below_lb.^2) + sum(above_ub.^2);

if viol > 0
    Q = penalty * (1 + viol);
    return;
end

% Unpack estimated parameters: theta = [rho, sigma_epsilon, beta, amin]
rho           = theta(1);
sigma_epsilon = theta(2);
beta          = theta(3);
amin          = theta(4);

% Assemble full 9-element vector expected by compute_moments:
% [beta, sigma, alpha, delta, amin, amax, n, rho, sigma_epsilon]
full_params = [beta, fixed.sigma, fixed.alpha, fixed.delta, amin, fixed.amax, fixed.n, rho, sigma_epsilon];

% Evaluate model moments


% 3. Evaluate model moments with safeguard against solver/simulation errors
try
    m_model = compute_moments(full_params, seed);
catch ME
    warning('SMM:EvalError', 'Model evaluation failed: %s', ME.message);
    Q = penalty;
    return;
end

m_model = m_model(:);
if length(m_model) ~= length(m_data) || any(~isfinite(m_model))
    Q = penalty;
    return;
end

% 4. Compute SMM criterion Q(theta) = (m_data - m_model)' * W * (m_data - m_model)
diff_m = m_data - m_model;
Q = diff_m' * W * diff_m;

if ~isfinite(Q)
    Q = penalty;
end
end