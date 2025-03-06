%% Load Data and Form Return Series
clear all; close all; clc; format shortG
load dataset.txt
data=flipud(dataset);
ret=log(data(2:end,:)./data(1:end-1,:))
stockNames = {'AAPL','MSFT','IBM','NVDA','GOOGL','AMZN'};
%% Statistical Analysis
% Descriptive Statistics
mu     = mean(ret)
sigma  = std(ret)
skew   = skewness(ret)
kurt   = kurtosis(ret)
% Time Series Plot
for i = 1:size(ret,2)
    figure
    plot(ret(:,i))
    title(['Return Series for stock ', stockNames{i}])
    xlabel('Time')
    ylabel('Return')
    grid on
% Histogram of Returns
    figure
    histogram(ret(:,i), 50)
    title(['Histogram of Returns ', stockNames{i}])
    xlabel('Return')
    ylabel('Frequency')
    grid on
% Q-Q Plot (Normality Check)
    figure
    qqplot(ret(:,i))
    title(['Q-Q Plot of Returns', stockNames{i}])
% Autocorrelation function (ACF)
    figure
    autocorr(ret(:,i))
    title(['Autocorrelation of Returns', stockNames{i}])
% Normality Test (Jarque-Bera Test)
% h = 0 indicates that we cannot reject the null hypothesis of normality
    [h(i), pVal(i), JBstat(i), critVal(i)] = jbtest(ret(:,i));
    if h(i) == 0
        fprintf('Jarque-Bera Test: Cannot reject normality (p = %f)\n', pVal);
    else
        fprintf('Jarque-Bera Test: Reject normality (p = %f)\n', pVal);
    end
end
%%
clc
jbtest(ret(:,1))
jbtest(ret(:,2))
jbtest(ret(:,3))
jbtest(ret(:,4))
jbtest(ret(:,5))
jbtest(ret(:,6))