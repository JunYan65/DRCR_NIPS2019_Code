k = 7
K = 100

l1_loss = zeros(K,k);
l2_loss = zeros(K,k);

for iter = 1:K
disp(iter)
disp(100)
    for j = 1:k
        n = 50*j;
        d = 5;
        X = randn(d, n);
        % target function f=|x_1|+...+|x_d|
        Y_star = sum(abs(X),1);
        Y = Y_star + 0.2 * randn(1,n);
        Ys = Y;
        % bandwidth %
        %h = 0.3*n^(-1/(5+d-1))*ones(1,d);
        %h=n^(-1/5)*[1;1;1;1;1]; 
        %a = [1;1;1;1;1]
        marki = 0;
        markloss = 1000;
        for ci = 1:100
            h = (ci/100)*n^(-1/(5+d-1))*ones(1,d);
            for i=1:n
                    Ys(i)=gaussian_kern_reg(X(:,i),X,Y,h); % Prediction
            end
            loss_cv = sumsqr(Ys - Y_star)/n;
            if loss_cv < markloss
                marki = ci;
                markloss = loss_cv;
            end
        end
        %disp(marki);
        h = (marki/100)*n^(-1/(5+d-1))*ones(1,d);
        for i=1:n
                Ys(i)=gaussian_kern_reg(X(:,i),X,Y,h); % Prediction
        end
        loss_1 = sum(abs(Ys - Y_star))/n;
        loss_2 = sqrt(sumsqr(abs(Ys - Y_star))/n);
        l1_loss(iter,j) = loss_1;
        l2_loss(iter,j) = loss_2;
    end
end
l1_loss
l2_loss
mean(l1_loss)
m2l = [0.1915    0.1944    0.1943    0.1921    0.1924    0.1903    0.1909]
m1l = [0.1534    0.1561    0.1552    0.1534    0.1538    0.1521    0.1520]



x = [50, 100, 150, 200, 250, 300, 350]
l1l1 = [0.14926217, 0.14537281, 0.14506484, 0.14070295, 0.13990859, 0.13723789,	0.136523256]
l2l1 = [0.1598674 , 0.15819003, 0.15395384, 0.1528812 , 0.1477016 , 0.1456073 ,	0.143583161]
badl1 = [2.012463883624774907e-01,2.049582042913922764e-01,2.050218559933872919e-01,2.138955366218921739e-01,2.118380016184172598e-01,2.118561986853172152e-01,2.133389511577850617e-01]

hold on

plot(x, l1l1, 'b', 'LineWidth',4)
plot(x, l2l1, 'r-.', 'LineWidth',4)

plot(x, badl1, 'black:', 'LineWidth',4)

plot(x, m1l, 'g--', 'LineWidth',4)
title('Empirical l_1 Loss','fontsize',18)
xlabel('n','fontsize',18)
ylabel('Loss','fontsize',18)
axis([50 350 0.12 0.28])
set(gca,'FontSize',12)
legend({'DRCR','LSE_{10}','LSE_{0.8}','Kernel'},'fontsize',15)





x = [50, 100, 150, 200, 250, 300, 350]
l1l2 = [0.18578856, 0.18254937, 0.18162452, 0.17632103, 0.17554751, 0.17245254,	0.172363809723]
l2l2 = [0.19867521, 0.19746269, 0.1933927 , 0.19201725, 0.18560912,      0.18350755,	0.180677155]
badl2 = [2.517635057812228627e-01,2.576192425112116546e-01,2.567770899472978097e-01,2.675852501337211420e-01,2.659805738279987786e-01,2.661463127501673132e-01,2.683495756941414734e-01]

hold on
plot(x, l1l2, 'b', 'LineWidth',4)
plot(x, l2l2, 'r-.', 'LineWidth',4)
plot(x, badl2, 'black:', 'LineWidth',4)
plot(x, mean(l2_loss), 'g--', 'LineWidth',4)

title('Empirical l_2 Loss','fontsize',18)
xlabel('n','fontsize',18)
ylabel('Loss','fontsize',18)
axis([50 350 0.12 0.28])
set(gca,'FontSize',12)
legend({'DRCR','LSE_{10}','LSE_{0.8}','Kernel'},'fontsize',15)


