function P = bondPrice(y, faceValue, couponRate, maturity)
    % bondPrice computes the price of a standard coupon bond
    %
    %   P = bondPrice(y, faceValue, couponRate, maturity)
    %
    % Inputs:
    %   y           = yield to maturity (annual)
    %   faceValue   = face (par) value of the bond
    %   couponRate  = annual coupon rate, e.g. 0.05 for 5%
    %   maturity    = time in years to maturity
    %
    % Output:
    %   P = bond price

    % Number of coupon payments per year (assume annual here)
    freq = 1; 
    % The coupon payment in currency units
    C = faceValue * couponRate/freq; 

    % Discount each coupon plus redemption at the end:
    P = 0;
    for t = 1:freq*maturity
        P = P + C / (1+y)^(t);
    end
    % Add face value redemption at final maturity:
    P = P + faceValue / (1+y)^(freq*maturity);
end


% Bond with 10-year maturity w/ annual coupon payments
faceValue   = 100;
couponRate  = 0.05;      % 5%
maturity    = 10;        % 10 years
P_market    = 99;


% Compute the yield to maturity (YTM) for this bond, under the assumption
% that the daily fluctuations in YTM follow independent and identically
% distributed (i.i.d) Gaussian random variables with a mean of 0 and a
% standard deviation of 0.006.

% Define a "wrapper" that is zero when the price is 99
f = @(y) bondPrice(y, faceValue, couponRate, maturity) - P_market;

% Use fsolve or fzero. We need an initial guess (e.g. y ~ 0.05).
yGuess = 0.05;  % just pick something reasonable
options = optimset('Display','off'); 

y0 = fzero(f, yGuess, options);
fprintf('Current yield that prices the bond at 99 is %.4f (%.2f%%)\n', ...
         y0, 100*y0);


% Estimate the probability of a 10% decline in the bond price within a
% 30-day period
P0 = bondPrice(y0, faceValue, couponRate, maturity);
thresholdPrice = 0.90 * P0;

% Solve for the yield yStar such that bondPrice(yStar) = thresholdPrice
f_threshold = @(y) bondPrice(y, faceValue, couponRate, maturity) ...
                   - thresholdPrice;
yStar = fzero(f_threshold, y0);

% Probability that the bond price falls below 90% is 
% Probability(yield >= yStar) if yStar > y0, or 
% Probability(yield <= yStar) if yStar < y0.
% In practice, for a standard coupon bond, yStar > y0 if price is dropping.

sigmaDaily = 0.006;       % daily stdev
sigma30    = sigmaDaily * sqrt(30);

% We want:  y_30 >= yStar  => P(Y >= yStar)
prob10pctDrop = 1 - normcdf(yStar, y0, sigma30);

fprintf('Probability of a 10%% price drop in 30 days: %.4f%%\n', ...
        100*prob10pctDrop);


% Compute the Value at Risk (VaR) for your bond at 99% confidence level
% across various horizons (1, 10, 20, 30, ..., 90 days) using the following
% methods

horizons = [1 10 20 30 40 50 60 70 80 90];
numHorizons = length(horizons);
alpha    = 0.99;  % 99% confidence => 1% tail


% 3.1 - Exact “Full” Formula via Direct Price Computation

% Pre-allocate array for VaR
VaR_exact_all = zeros(numHorizons,1);

z_99 = norminv(alpha);  % about 2.3263

for i = 1:numHorizons
    t = horizons(i);
    sigma_t = sigmaDaily * sqrt(t);
    y_quantile = y0 + z_99*sigma_t;
    P_quantile = bondPrice(y_quantile, faceValue, couponRate, maturity);
    VaR_exact  = P0 - P_quantile;

    % Store in array
    VaR_exact_all(i) = VaR_exact;

    fprintf('Exact VaR (inversion) at 99%% for %2d days = %.4f\n', ...
            t, VaR_exact);
end


% 3.2 Delta Approximation

% Pre-allocate array for VaR
VaR_delta_all = zeros(numHorizons,1);

function dPdy = bondPriceDerivative(y, faceValue, couponRate, maturity)
    % Finite difference for derivative:
    eps = 1e-6;
    P_plus  = bondPrice(y+eps, faceValue, couponRate, maturity);
    P_minus = bondPrice(y-eps, faceValue, couponRate, maturity);
    dPdy    = (P_plus - P_minus)/(2*eps);
end

% Then in main code:
dPdy_0 = bondPriceDerivative(y0, faceValue, couponRate, maturity);

for i = 1:numHorizons
    t = horizons(i);

    sigma_t = sigmaDaily * sqrt(t);
    VaR_delta = - dPdy_0 * (z_99 * sigma_t);
       
    % Store in array
    VaR_delta_all(i) = VaR_delta;

    fprintf('Delta VaR at 99%% for %2d days = %.4f\n', t, VaR_delta);
end


% 3.3 Delta‐Gamma Approximation

% Pre-allocate array for VaR
VaR_deltaGamma_all = zeros(numHorizons,1);

function d2Pdy2 = bondPriceSecondDerivative(y, faceValue, couponRate, maturity)
    eps = 1e-6;
    fPlus  = bondPriceDerivative(y+eps, faceValue, couponRate, maturity);
    fMinus = bondPriceDerivative(y-eps, faceValue, couponRate, maturity);
    d2Pdy2 = (fPlus - fMinus)/(2*eps);
end

gamma_0 = bondPriceSecondDerivative(y0, faceValue, couponRate, maturity);

for i = 1:numHorizons
    
    t = horizons(i);
    sigma_t = sigmaDaily*sqrt(t);
    % The random variable is N(0, sigma_t^2).
    % Under a worst-case shift of +z_99*sigma_t, approximate the price change:
    dP_approx = dPdy_0*(z_99*sigma_t) + 0.5*gamma_0*((z_99*sigma_t)^2);
    % Price goes down if yield goes up, so the 'loss' is -dP_approx:
    VaR_deltaGamma = - dP_approx;
    
    % Store in array
    VaR_deltaGamma_all(i) = VaR_deltaGamma;

    fprintf('Delta-Gamma VaR at 99%%, %2d days = %.4f\n', t, VaR_deltaGamma);
end


nSim = 10000;
rng(123);  % for reproducibility

% 3.4 Monte Carlo simulation with delta approximation

% Pre-allocate arrays for VaR and ES:
VaR_mc_delta_all = zeros(numHorizons,1);
ES_mc_delta_all = zeros(numHorizons,1);

for i = 1:numHorizons

    t = horizons(i);
    sigma_t = sigmaDaily*sqrt(t);
    dY = sigma_t*randn(nSim,1);  % 10,000 draws from N(0, sigma_t^2)
    Y_sim = y0 + dY;
    
    % Revalue each scenario:
    P_sim = P0 + dPdy_0 .* dY; 
    
    % The loss is L = P0 - P_sim
    L_sim = P0 - P_sim;
    
    % 99% VaR is the 99th percentile:
    VaR_mc_delta = quantile(L_sim, alpha);
    fprintf('Monte Carlo Full Reval 99%% VaR, %2d days: %.4f\n', t, VaR_mc_delta);
    
    % Similarly, we can compute the Expected Shortfall:
    lossesSorted = sort(L_sim);
    cutoffIndex  = ceil(alpha*nSim);
    % average of the worst (1-alpha)*nSim losses:
    ES_mc_delta = mean(lossesSorted(cutoffIndex:end));

    % Store in arrays:
    VaR_mc_delta_all(i) = VaR_mc_delta;
    ES_mc_delta_all(i) = ES_mc_delta;

    fprintf('Monte Carlo Full Reval 99%% ES,   %2d days: %.4f\n', t, ES_mc_delta);
end


% 3.5 Monte Carlo simulation with delta-gamma approximation.

% Pre-allocate arrays for VaR and ES:
VaR_mc_deltaGamma_all = zeros(numHorizons,1);
ES_mc_deltaGamma_all = zeros(numHorizons,1);

for i = 1:numHorizons

    t = horizons(i);
    sigma_t = sigmaDaily*sqrt(t);
    dY = sigma_t*randn(nSim,1);  % 10,000 draws from N(0, sigma_t^2)
    Y_sim = y0 + dY;
    
    % Revalue each scenario:
    P_sim = P0 + dPdy_0 .* dY + 0.5 * gamma_0 .* (dY.^2); 
    
    % The loss is L = P0 - P_sim
    L_sim = P0 - P_sim;
    
    % 99% VaR is the 99th percentile:
    VaR_mc_deltaGamma = quantile(L_sim, alpha);
    fprintf('Monte Carlo Full Reval 99%% VaR, %2d days: %.4f\n', t, VaR_mc_deltaGamma);
    
    % Similarly, we can compute the Expected Shortfall:
    lossesSorted = sort(L_sim);
    cutoffIndex  = ceil(alpha*nSim);
    % average of the worst (1-alpha)*nSim losses:
    ES_mc_deltaGamma = mean(lossesSorted(cutoffIndex:end));

    % Store in arrays:
    VaR_mc_deltaGamma_all(i) = VaR_mc_deltaGamma;
    ES_mc_deltaGamma_all(i) = ES_mc_deltaGamma;

    fprintf('Monte Carlo Full Reval 99%% ES,   %2d days: %.4f\n', t, ES_mc_deltaGamma);
end

% 3.6 Full Revaluation

% Pre-allocate arrays for VaR and ES:
VaR_mc_full_all = zeros(numHorizons,1);
ES_mc_full_all = zeros(numHorizons,1);

for i = 1:numHorizons

    t = horizons(i);
    sigma_t = sigmaDaily*sqrt(t);
    dY = sigma_t*randn(nSim,1);  % 10,000 draws from N(0, sigma_t^2)
    Y_sim = y0 + dY;
    
    % Revalue each scenario:
    P_sim = arrayfun(@(y) bondPrice(y, faceValue, couponRate, maturity), Y_sim);
    
    % The loss is L = P0 - P_sim
    L_sim = P0 - P_sim;
    
    % 99% VaR is the 99th percentile:
    VaR_mc_full = quantile(L_sim, alpha);
    fprintf('Monte Carlo Full Reval 99%% VaR, %2d days: %.4f\n', t, VaR_mc_full);
    
    % Similarly, we can compute the Expected Shortfall:
    lossesSorted = sort(L_sim);
    cutoffIndex  = ceil(alpha*nSim);
    % average of the worst (1-alpha)*nSim losses:
    ES_mc_full = mean(lossesSorted(cutoffIndex:end));

    % Store in arrays:
    VaR_mc_full_all(i) = VaR_mc_full;
    ES_mc_full_all(i) = ES_mc_full;

    fprintf('Monte Carlo Full Reval 99%% ES,   %2d days: %.4f\n', t, ES_mc_full);
end

figure;
plot(horizons, VaR_exact_all,         '-o', 'LineWidth',1.5, 'DisplayName','Exact'); hold on;
plot(horizons, VaR_delta_all,         '-s', 'LineWidth',1.5, 'DisplayName','Delta');
plot(horizons, VaR_deltaGamma_all,    '-^', 'LineWidth',1.5, 'DisplayName','Delta-Gamma');
plot(horizons, VaR_mc_delta_all,      '--o', 'LineWidth',1.5, 'DisplayName','MC Delta');
plot(horizons, VaR_mc_deltaGamma_all, '--s', 'LineWidth',1.5, 'DisplayName','MC Delta-Gamma');
plot(horizons, VaR_mc_full_all,       '--^', 'LineWidth',1.5, 'DisplayName','MC Full');

xlabel('Horizon (days)');
ylabel('VaR at 99%');
title('Comparison of VaR Methods');
legend('Location','best');
grid on;

