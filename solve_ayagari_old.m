% Problem 1 - part 3.

% Runing the code will release two figures: 
% Figure 1: V(k) and G(k) plots;
% Figure 2: capital stocks sequences starting from high and low k1

clear all
clc
close all
format compact

%Model Parameters
beta = 0.95;            % discount factor
sigma = 2;              % risk aversion
alpha = 0.7;            % capital share
delta = 0.05;           % depreciation rate
amin = 0;               % borrowing limit
amax = 10;              % maximum assets
n = 500;                % number of grid points
        
penalty = -10^(15);     % for utility function penalization of negative numbers
tolv=10^(-7);           % tolerance for value function iteration

Pi=[0.7 0.1 0.2 0;
    0.1 0.7 0.1 0.1;
    0.1 0.1 0.7 0.1;
    0 0.2 0.1 0.7;];  % Markov process transition matrix

e_vals=[0.5 0.9 1.4 2]; % Array of productivity values

P=(Pi)^10000;           % stationary distribution

Pit=Pi';
lt=length(e_vals);

%Steady state expected productivity shock
Ee_vals=P(1,:)*e_vals'; 

% Grid for assets
A = linspace(amin, amax, n);

r = 0.01;               % interest rate
w = 1;                  % wage

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



% % Plotting results
% figure;
% subplot(1,2,1);
% plot(A, V1); title('Value Functions'); legend('Low Shock', 'High Shock');
% subplot(1,2,2);
% plot(A, G); hold on; plot(A, A, '--k'); title('Policy Functions');


% % --- Simulation Parameters ---
% T = 20;             
% initial_asset_idx = round(n/2); 
% initial_shock_idx = 1; % Starting state index (1 or 2)
% 
% % Generate the sequence of shock INDICES (1s and 2s)
% % Note: markovchain(Trans, T, s0, me)
% shock_indices = markovchain(Pi, T, initial_shock_idx);
% 
% % --- Initialize Simulation Vectors ---
% s_path = zeros(T, 1);
% a_path = zeros(T+1, 1); 
% c_path = zeros(T, 1);
% 
% % Set initial asset state
% curr_a_idx = initial_asset_idx;
% 
% % --- Simulation Loop ---
% for t = 1:T
%     % 1. Get current shock index from the pre-generated path
%     curr_s_idx = shock_indices(t);
% 
%     % 2. Record values for plotting
%     s_path(t) = e_vals(curr_s_idx); % Map index (1,2) to value (0.9, 1.1)
%     a_path(t) = A(curr_a_idx);
% 
%     % 3. Use policy function to find next period's asset
%     next_a_idx = Pol_indices(curr_a_idx, curr_s_idx);
%     a_path(t+1) = A(next_a_idx);
% 
%     % 4. Calculate consumption
%     c_path(t) = (1+r)*a_path(t) + w*s_path(t) - a_path(t+1);
% 
%     % 5. Update asset index for next period
%     curr_a_idx = next_a_idx;
% end
% 
% % --- Plotting the Realizations ---
% time = 1:T;
% 
% figure;
% subplot(3,1,1);
% stairs(time, s_path, 'r', 'LineWidth', 1.5);
% ylabel('Income Shock (s)');
% title('20-Period Simulation of a Single Household');
% grid on;
% 
% subplot(3,1,2);
% plot(time, a_path(1:T), 'b-o', 'LineWidth', 1.5);
% ylabel('Assets (a)');
% grid on;
% 
% subplot(3,1,3);
% plot(time, c_path, 'g-s', 'LineWidth', 1.5);
% xlabel('Time Period');
% ylabel('Consumption (c)');
% grid on;