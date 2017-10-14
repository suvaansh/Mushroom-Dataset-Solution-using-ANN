function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections

W = zeros(L_out, 1 + L_in);

INIT_EPSILON = (sqrt(6)) / (sqrt(L_in + L_out));
W = (rand(L_out,L_in+1) * 2 * INIT_EPSILON) - INIT_EPSILON;

end
