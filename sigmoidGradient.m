function g = sigmoidGradient(z)   %g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function evaluated at z. 

g = zeros(size(z));

az = sigmoid(z);
%az = [ones(size(az,1),1) az];

g = (az.*(1-az));

end