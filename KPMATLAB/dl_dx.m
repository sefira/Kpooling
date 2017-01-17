function [code,mangrad,numgrad] = dl_dx(varargin)
% Output the final encode result
% Compute the number gradient and manual gradient 
% to compare these, if equal, then porve derivation formula is right
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
        % x = [0.2511;
        %     0.6160;
        %     0.4733;
        %     0.3517];
        % y = [0.2511;
        %     0.6160;
        %     0.4733;
        %     0.3517];
        sigma = 3;
    end
    % code
    [code,~] = forward(x,y,sigma);
   
    % number gradient
    n = size(x,1);
    EPSILON=1e-6;
    % gradient of x or y
    needxory = 'y';
    if(needxory == 'x')
        numgrad = zeros(1,size(x,1));
        for i=1:n
            d=zeros(n,1);
            d(i) = EPSILON;
            [~,a] = forward(x+d/2,y,sigma);
            [~,b] = forward(x-d/2,y,sigma);
            numgrad(i) = (a-b)/EPSILON;
        end
    else
        numgrad = zeros(size(x,1),1);
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

function [bilinear,loss] = forward(x,y,sigma)
% Origin mapping function
    % bilinear forward
    bilinear = x'*y;
    % f:loss: mean square loss, suppose target equal zeros
    loss = lossfunction(bilinear);
end

function loss = lossfunction(c)
    loss = (1/2)*(sum(c.^2));
end

function mangrad = backward(x,y,sigma,needxory)
    bilinear = forward(x,y,sigma);
    mangrad_l_f = l_f(bilinear);
    if(needxory == 'x')
        mangrad = mangrad_l_f*y';        
    else
        mangrad = mangrad_l_f*x;
    end
end

% the loss function
function mangrad = l_f(c)
    mangrad = c;
end
