% Problem 1 - part 3.

% Runing the code will release two figures: 
% Figure 1: V(k) and G(k) plots;
% Figure 2: capital stocks sequences starting from high and low k1

clear all
clc
close all
format compact

%Model Parameters

z=1;
beta = 0.95;
r = 0.01;
w = 1;
b = 0; %borrow constraint

n = 500;
amin = b;
amax = 10;
penalty = -10^(15);
tolv=10^(-7);


% Markov process
    Pi=[0.5 0.5;
       0.5 0.5];
   teta=[0.5 2];
P=(Pi)^10000;%stationary distribution
Pit=Pi';
lt=length(teta);
%Steady state
Eteta=P(1,:)*teta';

% Grid for assets
A = linspace(amin, amax, n);

% Matrix C of consumption with all possible combinations of K_t and K_t+1
C = zeros(n,n);

for t=1:lt
    for i = 1:n
        for j = 1:n
            C(i, j, t) = (1+r)*A(i) + teta(t)*w - A(j);
        end
    end
end

% Matrix U. Flow utility associated with consumption choices

U = zeros(n,n);

for t=1:lt
    for i = 1:n
        for j = 1:n
            if C(i, j, t) > 0
                U(i, j, t) = log(C(i, j, t));
                else
                U(i, j, t) = penalty; % Assign penalty for non-positive values
            end
        end
    end
end
% First guess of value function V(k) = 0

V0 = ones(n,lt);
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
        % We use repmat or broadcasting to add EV to each row of U
        obj = U(:,:,t) + beta * EV(:,t)'; 
        
        [V1(:,t), Pol_indices(:,t)] = max(obj, [], 2);
    end
    
    % Step 3: Check convergence (Supremum norm)
    dist = max(max(abs(V1 - 0)));
    
    % Update
    V0 = V1;
    
    if mod(iteration, 50) == 0
        fprintf('Iteration %d, Dist: %e\n', iteration, dist);
    end
    iteration = iteration + 1;
end










% 
% while norm(V1-V0)>tolv && iteration < maxiter
%     V0=V1;
%     for t=1:lt % teta
%         for i=1:n % cap today
%             V1(i,t)=max(U(:,i,t)+beta*V0*Pit(:,t));
%         end
%     end
% 
%     % Continue to next iter
%     iteration = iteration +1;
% 
% end
% 
% %Compute the optimal policy functions
% optim=[];
%  for t=1:lt
%    for i=1:n
%       [x optim(i,t)]=max(U(:,i,t)+beta*V1*Pit(:,t));
%    end
%  end

G=A(Pol_indices);


% Plotting results
figure;
subplot(1,2,1);
plot(A, V1); title('Value Functions'); legend('Low Shock', 'High Shock');
subplot(1,2,2);
plot(A, G); hold on; plot(A, A, '--k'); title('Policy Functions');


% --- Simulation Parameters ---
T = 20;             
initial_asset_idx = round(n/2); 
initial_shock_idx = 1; % Starting state index (1 or 2)

% Generate the sequence of shock INDICES (1s and 2s)
% Note: markovchain(Trans, T, s0, me)
shock_indices = markovchain(Pi, T, initial_shock_idx);

% --- Initialize Simulation Vectors ---
s_path = zeros(T, 1);
a_path = zeros(T+1, 1); 
c_path = zeros(T, 1);

% Set initial asset state
curr_a_idx = initial_asset_idx;

% --- Simulation Loop ---
for t = 1:T
    % 1. Get current shock index from the pre-generated path
    curr_s_idx = shock_indices(t);
    
    % 2. Record values for plotting
    s_path(t) = teta(curr_s_idx); % Map index (1,2) to value (0.9, 1.1)
    a_path(t) = A(curr_a_idx);
    
    % 3. Use policy function to find next period's asset
    next_a_idx = Pol_indices(curr_a_idx, curr_s_idx);
    a_path(t+1) = A(next_a_idx);
    
    % 4. Calculate consumption
    c_path(t) = (1+r)*a_path(t) + w*s_path(t) - a_path(t+1);
    
    % 5. Update asset index for next period
    curr_a_idx = next_a_idx;
end

% --- Plotting the Realizations ---
time = 1:T;

figure;
subplot(3,1,1);
stairs(time, s_path, 'r', 'LineWidth', 1.5);
ylabel('Income Shock (s)');
title('20-Period Simulation of a Single Household');
grid on;

subplot(3,1,2);
plot(time, a_path(1:T), 'b-o', 'LineWidth', 1.5);
ylabel('Assets (a)');
grid on;

subplot(3,1,3);
plot(time, c_path, 'g-s', 'LineWidth', 1.5);
xlabel('Time Period');
ylabel('Consumption (c)');
grid on;