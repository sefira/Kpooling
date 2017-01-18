function [output,mangrad,numgrad] = dl_dx(varargin)
% Output the final output
% Compute the manual gradient and number gradient to compare them
% if they are equal, then the derivation formula should be right
    global x;
    global y;
    global sigma;
    if nargin >= 3
        x = varargin{1};
        y = varargin{2};
        sigma = varargin{3};
    else
        x = rand(5,1);
        y = rand(5,1);
        % x = [0.6476;
        %     0.6790;
        %     0.6358;
        %     0.9452;
        %     0.2089];
        % y = [0.7093;
        %     0.2362;
        %     0.1194;
        %     0.6073;
        %     0.4501];
        sigma = 10;
    end
    % output
    [output,~] = forward(x,y,sigma);
   
    % number gradient
    n = size(x,1);
    numgrad = zeros(n,1);
    EPSILON=1e-6;
    % gradient of x or y
    needxory = 'x';
    if(needxory == 'x')
        for i=1:n
            d=zeros(n,1);
            d(i) = EPSILON;
            [~,a] = forward(x+d/2,y,sigma);
            [~,b] = forward(x-d/2,y,sigma);
            numgrad(i) = (a-b)/EPSILON;
        end
    else
        for i=1:n
            d=zeros(n,1);
            d(i) = EPSILON;
            [~,a] = forward(x,y+d/2,sigma);
            [~,b] = forward(x,y-d/2,sigma);
            numgrad(i) = (a-b)/EPSILON;
        end
    end
    % manual gradient, computed by derivation
    mangrad = backward(x,y,sigma,needxory);
end

function [RBF,loss] = forward(x,y,sigma)
% Origin mapping function
    dist = x-y;
    dist = dist'*dist;
    dist = -dist / (2*(sigma^2));
    RBF = exp(dist);
    % MSE loss, suppose target equals zeros
    loss = lossfunction(RBF);
end

function loss = lossfunction(x)
    loss = (1/2)*(x'*x);
end

function mangrad = backward(x,y,sigma,needxory)
    output = forward(x,y,sigma);
    mangrad_l_f = l_f(output);
    output = output / (-sigma^2);
    if(needxory == 'x')
        mangrad = mangrad_l_f*output*(x-y);
    else
        mangrad = mangrad_l_f*output*(y-x);
    end
end

% the loss function
function mangrad = l_f(x)
    mangrad = x;
end
