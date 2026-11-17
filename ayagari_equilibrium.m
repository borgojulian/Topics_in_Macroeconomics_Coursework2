function [K_star, r_star, value_function, policy_function, stationary_distribution, mean_assets, it] = ayagari_equilibrium(beta,sigma,alpha,delta,amin,amax,n,e_vals,Pi)
% Inputs as requested
% Outputs: equilibrium aggregate capital, interest rate, mean assets, iterations

% Parameters for the outer loop
phi = 0.9;           % damping parameter (choose appropriate)
tolK = 1e-6;         % tolerance for K fixed point (can reuse VFI tol)
maxit = 200;

% Initial guess for K
K = 1.0;             % choose sensible initial guess or pass as argument
it = 0;
err = Inf;

% Asset grid (example)
asset_grid = linspace(amin,amax,n)';

while err > tolK && it < maxit
    it = it + 1;

    % 1) compute interest rate associated with current K
    r = alpha * K^(alpha-1) - delta;

    % 2) Solve household problem given r (and other params)
    %    You can move your current VFI/code into a function:
    %    [policy_a, policy_l, V] = solve_household(r, beta, sigma, a_grid, e_vals, Pi, ...);
    [value_function, policy_function, stationary_distribution, mean_assets] = solve_household_VFI(r, beta, sigma, asset_grid, e_vals, Pi,n);

    % 3) Compute stationary distribution given policy function
    %    [dist] = stationary_dist(policy_a, a_grid, e_vals, Pi);
    %dist = stationary_dist(policy_a, a_grid, e_vals, Pi);

    % 4) Compute mean assets implied by stationary distribution
    %mean_a = sum( (a_grid') .* sum(dist,1) ); % adjust orientation if needed
    % If dist is size (n x m) with rows assets and cols income states:
    % mean_a = sum(a_grid .* sum(dist,2));

    % 5) Update aggregate capital guess with damping
    K_new = phi * K + (1-phi) * mean_a;

    err = abs(K_new - K);
    K = K_new;
end

K_star = K;
r_star = alpha * K_star^(alpha-1) - delta;

end

%% 
function [value_function, policy_function, stationary_distribution, mean_assets] = solve_household_VFI(r, beta, sigma, asset_grid, e_vals, Pi,n)

penalty = -10^(15);     % for utility function penalization of negative numbers
tolv=10^(-7);           % tolerance for value function iteration
lt=length(e_vals);
A=asset_grid;
% Matrix C of consumption with all possible combinations of a_t and a_t+1
C = zeros(n,n,lt);

for t=1:lt
    for i = 1:n
        for j = 1:n
            C(i, j, t) = (1+r)*A(i) + e_vals(t)*w - A(j);
        end
    end
end

% Matrix U. Flow utility associated with consumption choices
U = zeros(n,n, lt);

%utility function
if sigma == 1
    u = @(c) log(max(c, 1e-10));
else
    u = @(c) (max(c, 1e-10).^(1-sigma) - 1) ./ (1-sigma);
end

%create the utility matrix
for t=1:lt
    for i = 1:n
        for j = 1:n
            if C(i, j, t) > 0
                U(i, j, t) = u(C(i, j, t));
                else
                U(i, j, t) = penalty;  % Assign penalty for non-positive values
            end
        end
    end
end

% Value function iteration to solve the household problem given r.


V0 = zeros(n,lt); % First guess of value function V(a,e) = 0 for all combinations of assets and shocks
V1 = zeros(n,lt); % New value function to be computed
Pol_indices = zeros(n, lt);

%setting initial values for iteration
iteration = 1;
maxiter = 500;
dist=10;

while dist > tolv && iteration < maxiter
    
    % Step 1: Compute Expected Value tomorrow for each shock today
    % EV(future_assets, current_shock)
    % EV = V * Pi' (if V is n x lt)
    EV = V0 * Pi'; 
    
    % Step 2: Maximize for each current shock state
    for t = 1:lt
        % Objective: Utility today + discounted expected value tomorrow
        obj = U(:,:,t) + beta * EV(:,t)'; %broadcasting repeat the 1xn vector EV(:,t)' adding it to each n row of the nxn matrix U(:,:,t). This way, choosing a_j tomorrow (columns) yields the same continuation value regardles of the current state assets today a_i (rows)
        
        [V1(:,t), Pol_indices(:,t)] = max(obj, [], 2);
    end
    
    % Step 3: Check convergence (Supremum norm)
    dist = max(max(abs(V1 - V0)));
    
    % Update
    V0 = V1;
    
    if mod(iteration, 50) == 0
        fprintf('Iteration %d, Dist: %e\n', iteration, dist);
    end
    iteration = iteration + 1;
end

%Policy function for assets next period
G=A(Pol_indices);

%Stationary distribution
N = n * lt;
Q = zeros(N, N);

for i = 1:n
    for t = 1:lt
        row = (i - 1) * lt + t;              
        i_next = Pol_indices(i, t);          % Next period's chosen asset index
        
        for t_next = 1:lt
            col = (i_next - 1) * lt + t_next;
            Q(row, col) = Pi(t, t_next);    
        end
    end
end
% Q is the Markov chain on the household's state vector. 
% It shows the probability of transitioning from a pair of assets and 
% shock today (a_t,e_t) to a pair (a_t+1 , e_t+1) tomorrow, defined by the 
% policy function G and the Markov process of shocks Pi.

% Solve for the stationary invariant distribution of states (a,e): (I - Q' + ones(N,N)) * lambda_vec = ones(N,1)
A_mat = eye(N) - Q' + ones(N, N);
lambda_vec = A_mat \ ones(N, 1);

% A stationary markov chain (Q_stat,lambda_vec) satisfies (Q_stat' - I) * lambda_vec = 0
% where Q_stat is the stationary transition matrix and lambda_vec is the
% stationary distribution of states (is an eigenvector associated with a
% unit eigenvalue of Q_stat'. (Sargent and Ljunqvist's Recursive Macroeconomic Theory - Chapter 2 - 2.2.1.).
% 
% Here I check the stationary markov chain condition:
Q_stat = Q^10000;
tol = 1e-10;
residual = (Q_stat' - eye(N)) * lambda_vec;
difference = max(abs(residual));

if difference < tol
    fprintf('lambda_vec is stationary for Q_stat within tolerance.\n');
else
    fprintf('Maximum stationarity residual: %e\n', difference);
end

% Unstack back into an (n x lt) matrix
lambda = (reshape(lambda_vec, lt, n))';

% Aggregate capital supply
% Mean asset holdings under the stationary distribution.
% lambda(i,t) is the probability of being in state (A(i), e_vals(t)).
% G(i,t) is the asset choice for the next period from that state. lambda
% and G are both (n x lt) matrices.
% Element-wise multiplication (.*) weights each possible next-period
% asset choice by the probability of its corresponding current state. Sum
% over the element-wise multiplication is the weighted average of assets
% under the stationary distribution.
mean_assets = sum(lambda(:) .* G(:));
value_function = V1;
policy_function = G;
stationary_distribution = lambda;

