function [J grad] = nnCostFunction(nn_params, ...
input_layer_size, ...
hidden_layer_size, ...
num_labels, ...
X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
num_labels, (hidden_layer_size + 1));

% Dataset size
m = size(X, 1);

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Add biases
X = [ones(m,1) X];

% Inplement foward propagation
% a1 = X; 
a2 = sigmoid(Theta1 * X');
a2 = [ones(m,1) a2'];

a3 = sigmoid(Theta2 * a2');

% Reshape y labels from number 0-9 to vectors for multiclassification
yk = zeros(num_labels, m); 
for i=1:m,
    yk(y(i),i)=1;
end

% Implement Cost Function J without regularization
J = (1/m) * sum ( sum (  (-yk) .* log(a3)  -  (1-yk) .* log(1-a3) ));

% Delete biases for regularization
t1 = Theta1(:,2:size(Theta1,2));
t2 = Theta2(:,2:size(Theta2,2));

% Implement regularization formula
Reg = lambda  * (sum( sum ( t1.^ 2 )) + sum( sum ( t2.^ 2 ))) / (2*m);

% Implement Cost Function J with regularization
J = J + Reg;


% Implement Backpropagation Algorithm
for t=1:m,
  a1 = X(t,:); % 1x401
  z2 = Theta1 * a1'; % 25x401 * 401x1 =  number of layer 2 units x 1(25x1)
  a2 = sigmoid(z2);
	a2 = [1 ; a2]; % add bias

  z3 = Theta2 * a2; % 10x26 * 26x1 = 10x1
  a3 = sigmoid(z3); % final activation layer a3 == h(theta)
  
  z2=[1; z2]; % add bias to z2, 26x1
  delta_3 = a3 - yk(:,t); % 10x1
  delta_2 = (Theta2' * delta_3) .* sigmoidGradient(z2); % (26x10 * 10x1) .* 26x1 = 26x1
  
  delta_2 = delta_2(2:end); % Theta2_grad is 25x401, so delete the bias
  Theta2_grad = Theta2_grad + delta_3 * a2';
  Theta1_grad = Theta1_grad + delta_2 * a1;

end

% Compute partial derivatives
% j = 0
Theta1_grad(:, 1) = Theta1_grad(:, 1) ./ m;
% j != 0
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) ./ m + ((lambda/m) * Theta1(:, 2:end));

% j = 0
Theta2_grad(:, 1) = Theta2_grad(:, 1) ./ m;
% j != 0
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) ./ m + ((lambda/m) * Theta2(:, 2:end));


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
