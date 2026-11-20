function [cons_path, asset_path, income_path, shock_path] = simulate_model(e_vals,Pi,asset_grid,stationary_distribution,policy_function,w_star,r_star)

N_agents = 100000;
T = 500;
T_burn = 100;
lt = length(e_vals);

c_sim = zeros(N_agents,T);   % Path for consumption
a_sim = zeros(N_agents,T+1);   % Path for assets
w_sim = zeros(N_agents,T);   % Path for labor income
i_e_sim = zeros(N_agents,T); % Sequence of shocks (index 1,2,3,or 4)


%initialization of agents across (a,e) space
prob_vector = stationary_distribution(:);                       % 2000x1 probability vector
cdf = cumsum(prob_vector);
cdf(end) = 1; % Safeguard against floating point rounding
u = rand(N_agents,1);                                           % 100000 uniform draws in (0,1)
idx = discretize(u, [0; cdf]);                                  % 500 linear indices
[rows, cols] = ind2sub(size(stationary_distribution), idx);
a_sim(:,1) = asset_grid(rows);                                  % assign the sampled asset values
i_e_sim(:,1) = cols;                                            % assign the sampled shock indices


% Simulate Shocks 
Pi_cdf = [zeros(lt, 1), cumsum(Pi, 2)];
Pi_cdf(:, end) = 1;                         %  Safeguard against floating point rounding


U_shocks = rand(N_agents, T-1); % N_agents x T-1 matrix of random numbers between 0 and 1 from the uniform distribution
for t = 1:(T-1)
    for s = 1:lt
        % Find all agents currently in shock state 's' at time 't'
        mask = (i_e_sim(:, t) == s);
        if any(mask) % if no agent in shock 's', continue to next shock
            % Draw their next state using the s-th row of the transition CDF
            i_e_sim(mask, t+1) = discretize(U_shocks(mask, t), Pi_cdf(s, :));

            % discretize(X, edges) takes an array of continuous data X and a vector of bin boundaries 
            % edges (Pi_cdf(s, :) is the cumulative probability of each next period shock, conditional 
            % on being in state s in the current period), and returns an array of the same size of X 
            % containing the integer index of the bin each value fell into (index of next period shock).
        end
    end
end
e_sim = e_vals(i_e_sim); % assign the sampled shock values



% Create one makima interpolant per shock state for evaluating the policy
% function at arbitrary asset-grid values.
g_interp = cell(lt, 1);
for s = 1:lt
    g_interp{s} = griddedInterpolant(asset_grid, policy_function(:, s), ...
        'makima');
end


% Asset and Consumption Paths
for t = 1:T
    for s = 1:lt
        mask = (i_e_sim(:, t) == s);
        if any(mask)
            % Evaluate next period's assets using the interpolant
            a_next = g_interp{s}(a_sim(mask, t));
            
            % Enforce the borrowing constraint (makima may return
            % negative assets when interpolating close to the borrowing
            % constraint).
            a_sim(mask, t+1) = max(a_next, asset_grid(1)); 
        end
    end
    
    % Calculate consumption from the budget constraint
    c_sim(:, t) = (1 + r_star) * a_sim(:, t) + w_star * e_sim(:, t) - a_sim(:, t+1);
    
    % Calculate labor income path
    w_sim(:, t) = w_star * e_sim(:, t);
end

% --- 5. Discard Burn-in Periods ---
cons_path = c_sim(:, (T_burn + 1):T);
asset_path = a_sim(:, (T_burn + 1):T);
income_path = w_sim(:, (T_burn + 1):T);
shock_path = e_sim(:, (T_burn + 1):T);
end



