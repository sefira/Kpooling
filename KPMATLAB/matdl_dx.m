function [output,mangrad,numgrad] = matdl_dx(varargin)
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
        x = rand(5,3);
        y = rand(5,3);
        sigma = 10;
    end
    % output
    [output,~] = forward(x,y,sigma);
   
    % number gradient
    n = size(x,1);
    m = size(x,2);
    numgrad = zeros(n,m);
    EPSILON=1e-6;
    % gradient of x or y
    needxory = 'x';
    if(needxory == 'x')
        for i=1:n
            for j=1:m
                d=zeros(n,m);
                d(i,j) = EPSILON;
                dx1 = x + d/2;
                [~,a] = forward(dx1,y,sigma);
                dx2 = x - d/2;
                [~,b] = forward(dx2,y,sigma);
                dx1-dx2
                (a-b)
                numgrad(i,j) = (a-b)/EPSILON;
            end
        end
    else
        for i=1:n
            for j=1:m
                d=zeros(n,m);
                d(i,j) = EPSILON;
                [~,a] = forward(x,y+d/2,sigma);
                [~,b] = forward(x,y-d/2,sigma);
                numgrad(i,j) = (a-b)/EPSILON;
            end
        end
    end
    % manual gradient, computed by derivation
    mangrad = backward(x,y,sigma,needxory);
end

function [RBF,loss] = forward(x,y,sigma)
% Origin mapping function
    n = size(x,1);
    KMat = zeros(n,n);
    for i=1:n
        for j=1:n
            dist = x(i,:)-y(j,:);
            dist = dist*dist';
            dist = -dist / (2*(sigma^2));
            RBF = exp(dist);
            KMat(i,j) = RBF;
        end
    end
    % MSE loss, suppose target equals zeros
    loss = lossfunction(KMat);
end

function loss = lossfunction(x)
    loss = sum(sum((1/2)*(x.^2)));
end

function mangrad = backward(x,y,sigma,needxory)
    [output,~] = forward(x,y,sigma);
    mangrad_l_f = l_f(output);
    output = output / (-sigma^2);
    n = size(x,1);
    m = size(x,2);
    mangrad = zeros(n,m);
    if(needxory == 'x')
        for i=1:n
            for j=1:n
                mangrad(i,:) = mangrad(i,:) + mangrad_l_f*output*(x(i,:)-y(j,:));
            end
        end
    else
        for i=1:n
            for j=1:n
                mangrad(i,:) = mangrad(i,:) + mangrad_l_f*output*(y(i,:)-x(j,:));
            end
        end
    end
end

% the loss function
function mangrad = l_f(x)
    mangrad = x;
end
