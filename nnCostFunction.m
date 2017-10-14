function [J , grad]  = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification


Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


newy = zeros(m,num_labels);
for j=1:m
    newy(j,y(j)) = 1;
end;

X = [ones(m,1) X];
z2 = X*(Theta1');
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];
z3 = a2*(Theta2');
a3 = sigmoid(z3);
h = a3;
var1 = sum(sum((log(h)).*newy)) + sum(sum((1.-newy).*(log(1.-h))));
J = (-1/m)*(var1);

var2 = sum(sum((Theta1(:,(2:end))).^2)) + sum(sum((Theta2(:,(2:end))).^2));

reg = (lambda / (2*m))*var2;

J = J + reg;

delta3 = a3 - newy;
delta2 = delta3*Theta2 .* [ones(size(z2,1),1) sigmoidGradient(z2)];


%delta3 = h - newy;
%delta2 = (delta3 * Theta2) .* sigmoidGradient(z2);

DELTA1 = zeros(hidden_layer_size , (input_layer_size+1));
DELTA2 = zeros(num_labels , (hidden_layer_size+1));

DELTA2 = (delta3')*a2;
DELTA1 = (delta2(:,(2:end))')*X;

DELTA2 = DELTA2./m;
DELTA1 = DELTA1./m;

Theta1_grad = [DELTA1(:,1) (DELTA1(:,(2:end))+((lambda/(m))*(Theta1(:,(2:end)))))];
Theta2_grad = [DELTA2(:,1) (DELTA2(:,(2:end))+((lambda/(m))*(Theta2(:,(2:end)))))];


% Unrolling gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
