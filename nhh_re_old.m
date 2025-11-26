%---
%title: "NHH-Re"
%output:
%  pdf_document: default
%  html_document: default
%date: "2025-06-04"
%---

%```{r setup, include=FALSE}
%knitr::opts_chunk$set(echo = TRUE)


%```


%Modified t_j



%```{r}
%Remove all objects from the current workspace:
clear;

% Parameters
n = 30;                % Number of measurements / terms
%r = 3                 % Sobolev regularity
T = 5;               % Large T for error decay .... 
                      %Actually, we need small value to avoid overflow error
x0 = sqrt(2);           % Measurement point (should avoid rational multiples of pi)
x_grid = linspace(0, pi, 200); % Points to evaluate f_n(x), length.out=200+

% Define the measurement times t_k as in (4)
t_k = @(k, T) factorial(2 * k - 1) / ( 8^(k-1) * factorial(k) * factorial(k - 1) ) * T;


t_vec = arrayfun( @(k) t_k(k, T), 1:n);

% Simulate the true initial condition and its Fourier coefficients
f_true = @(x) sin(2 * x) + 0.5 * sin(5 * x);
f_hat_true = @(j) integral(@(x) f_true(x) .* sin(j * x), 0, pi) * 2 / pi;
% 2/pi * integral_0^pi f_true(x) * sin(j x) dx


% Simulate measurements u(x0, t_k) (homogeneous case: F = 0)
n_terms = 50;
u_xtk = @(x0, t_vec, n_terms) arrayfun(@(t) sum( arrayfun(@(j) exp(-j^2 * t) * f_hat_true(j) * sin(j * x0), 1:n_terms ) ) , t_vec); % Define Solution Representation for u(x_0,t_k) as in (1)  






u_data = u_xtk(x0, t_vec, n_terms);

% Recursive computation of approximate Fourier coefficients bar{f}_k
bar_f_hat = zeros(1, ceil(n / 2));
for k = 1:length(bar_f_hat)
  tk = t_vec(k);
  if k == 1
      bar_f_hat(1) = exp(tk) * u_data(1) / sin(x0);
  else
      sum_prev = sum(arrayfun(@(j) exp(-j^2 * tk) * bar_f_hat(j) * sin(j * x0), 1:(k-1))); % Define the recursive definition of fourier constant truncated approximation at k=1 as in (2)
      bar_f_hat(k) = exp(k^2 * tk) * (u_data(k) - sum_prev) / sin(k * x0);  % Define the recursive definition of fourier constant truncated approximation at k\geq2 as in (3)
  end
end



% Construct truncated series approximation bar{f}_n(x)
bar_f_n  = @(x) sum(arrayfun(@(k) bar_f_hat(k) * sin(k * x), 1:length(bar_f_hat)));
f_approx = @(x) arrayfun(bar_f_n,x);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot the true and reconstructed initial condition
fig1 = figure(1);
plot(x_grid,f_true(x_grid),"Color",[0 1 0],"LineStyle","-","LineWidth",2);hold on;
plot(x_grid,f_approx(x_grid),"Color",[1 0 0],"LineStyle","--","LineWidth",2);
xlabel('x');ylabel('f(x)');
title('Initial Condition Reconstruction');
legend('True f(x)', 'Reconstructed f_n(x)', 'Location', 'northeast');
ylim([min(f_true(x_grid)), max(f_true(x_grid))]);hold off;


%```
%linear sequence


%```{r}
%#Remove all objects from the current workspace:
clear;

% Parameters
n = 20;                % Number of measurements / terms
%r = 3                 % Sobolev regularity
t0 = 1e-3;               % Large T for error decay .... 
x0 = sqrt(2);           % Measurement point (should avoid rational multiples of pi)
x_grid = linspace(0, pi, 200); % Points to evaluate f_n(x), length.out=200+

% Define the measurement times t_k as in (4)
t_k = @(k, T) (n + k -1   ) * T;


t_vec = arrayfun( @(k) t_k(k, t0), 1:n);

% Simulate the true initial condition and its Fourier coefficients
f_true = @(x) sin(2 * x) + 0.5 * sin(5 * x);
f_hat_true = @(j) integral(@(x) f_true(x) .* sin(j * x), 0, pi) * 2 / pi;
% 2/pi * integral_0^pi f_true(x) * sin(j x) dx



% Simulate measurements u(x0, t_k) (homogeneous case: F = 0)
n_terms = 50;
u_xtk = @(x0, t_vec, n_terms) arrayfun(@(t) sum( arrayfun(@(j) exp(-j^2 * t) * f_hat_true(j) * sin(j * x0), 1:n_terms ) ) , t_vec); % Define Solution Representation for u(x_0,t_k) as in (1)  





u_data = u_xtk(x0, t_vec, n_terms);

% Recursive computation of approximate Fourier coefficients bar{f}_k
bar_f_hat = zeros(1, ceil(n / 2));
for k = 1:length(bar_f_hat)
  tk = t_vec(k);
  if k == 1
      bar_f_hat(1) = exp(tk) * u_data(1) / sin(x0);
  else
      sum_prev = sum(arrayfun(@(j) exp(-j^2 * tk) * bar_f_hat(j) * sin(j * x0), 1:(k-1))); % Define the recursive definition of fourier constant truncated approximation at k=1 as in (2)
      bar_f_hat(k) = exp(k^2 * tk) * (u_data(k) - sum_prev) / sin(k * x0);  % Define the recursive definition of fourier constant truncated approximation at k\geq2 as in (3)
  end
end



% Construct truncated series approximation bar{f}_n(x)
bar_f_n  = @(x) sum(arrayfun(@(k) bar_f_hat(k) * sin(k * x), 1:length(bar_f_hat)));
f_approx = @(x) arrayfun(bar_f_n,x);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot the true and reconstructed initial condition
fig2 = figure(2);
plot(x_grid,f_true(x_grid),"Color",[0 1 0],"LineStyle","-","LineWidth",2);hold on;
plot(x_grid,f_approx(x_grid),"Color",[1 0 0],"LineStyle","--","LineWidth",2);
xlabel('x');ylabel('f(x)');title('Initial Condition Reconstruction');
legend('True f(x)', 'Reconstructed f_n(x)', 'Location', 'northeast');
ylim([min(f_true(x_grid)), max(f_true(x_grid))]);hold off;
%```
%```