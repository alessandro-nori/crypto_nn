%%
clc
clear all
close all
%%

f = @(x) 1./(1+exp(-x)); % sigmoid function
x = -600:1:600;
y = f(x);
n = 1; % polynomial derivative degree
p_d = polyfit(x, y, n);

l = 200; % interval
xx = -l:1:l;
relu = max(0, xx);
yy = polyval(p_d, x);

p = [p_d 50]
yy2 = polyval(p, xx);
plot(xx, yy2, xx, relu)
legend('polynomial ReLU (degree 2)', 'ReLU');
