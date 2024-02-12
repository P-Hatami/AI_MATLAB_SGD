# AI_MATLAB_SGD
Stochastic Gradient Descent Algorithm Example
% Generate synthetic data
n = 100; % Number of data points
d = 2;   % Dimension of features
X = randn(n, d); % Random features
y = 2 * X(:, 1) - 3 * X(:, 2) + 0.5 * randn(n, 1); % Linear target variable

% Initialize parameters
w = zeros(d, 1); % Initial weight vector
learning_rate = 0.01; % Learning rate
max_epochs = 1000; % Maximum number of iterations

% Stochastic Gradient Descent
for epoch = 1:max_epochs
    for i = 1:n
        % Compute gradient for the i-th data point
        gradient = -X(i, :)' * (y(i) - X(i, :) * w);
        
        % Update weights
        w = w - learning_rate * gradient;
    end
    
    % Compute loss (mean squared error)
    loss = sum((X * w - y).^2) / n;
    
    % Display progress
    fprintf('Epoch %d: Loss = %.4f\n', epoch, loss);
end

% Final weight vector
fprintf('Final weights: w1 = %.4f, w2 = %.4f\n', w(1), w(2));
