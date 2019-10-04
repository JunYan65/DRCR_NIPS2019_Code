randn('state', 1);

d = 5;
n = 10;
X = randn(d, n);
% target function f=|x_1|+...+|x_d|
Y_star = sum(abs(X),1);
Y = Y_star + 0.3 * randn(1,n);

cvx_begin quiet
    variables Xi(d,n) g(1,n) C h(1,n)
    minimize sum(h)/n + n^(-2/5)*C
    subject to
        max(max(abs(Xi))) <= log(n);
        max(max(abs(Xi))) <= C;
        for i = 1 : n
            abs(Y(i) - g(i)) <= h(i);
        end
        for i = 1 : n
            for j = 1 : n
                g(i) >= g(j) + dot(Xi(:,j), X(:,i) - X(:,j))
            end
        end
cvx_end

l1_loss = sum(abs(Y_star - g))/n 
l2_loss = sqrt(sumsqr(Y_star - g)/n)

l1_loss
l2_loss
