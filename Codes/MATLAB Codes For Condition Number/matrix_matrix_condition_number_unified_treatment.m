%   Finding the worst case condition number for matrix-matrix scheme
%   We have n workers and s = sm - x stragglers.
%   Storage fraction gammaA = 1/kA and gammaB = 1/kB.
%   ellu is the number of uncoded blocks in each worker.
%   ellc is the number of coded blocks in each worker.
%   Worst condition number depends on the random matrices random_matA and random_matB.
%   Different simulations may provide different worst case condition
%   numbers and different worst choice of workers.
%   One can increase the number of trials to try to find a smaller worst case 
%   condition number.


clc
close all
clear

n = 24;                                                 % Number of workers
kA = 4;
kB = 5;
gammaA = 1/kA;
gammaB = 1/kB;
DeltaA = lcm(n,1/gammaA);
DeltaB = kB;
ell = DeltaA*gammaA;
Delta = DeltaA*DeltaB;
ellu = Delta/n;
ellc = ell - ellu;
k = kA*kB;
sm = n - k;

x = 0;                                           % One can set, x <= sm
s = sm - x;
threshold = k + x;
choices = combnk(1:n,threshold);
condition_no = zeros(nchoosek(n,threshold),1);

y = floor(kA/sm*x);
nz = DeltaA/kA;
nz1 = ellu+1;

denA = kA - y;                                   % density for coding of A
denB = floor(kB/2)+1;                            % density for coding of B
no_trials = 10;                                  % Number of trials

for trial = 1 : no_trials

    random_matA{trial} = randn(n*ellc,DeltaA);   % Random matrix for A
    ind = 0;
    for i = 1:n
        ee = zeros(1,DeltaA);
        ee((i-1)*DeltaA/n+1) = 1;
        for j = 1:ellu
            Coding_matrixA{i}(j,:)= ee;
            ee = circshift(ee,1);
        end
    end
    
    CM = zeros(ellc,n);
    CM(1,1) = nz1;
    if ellc > 1
        for i = 2 : ellc
            CM(i,1) = CM(i-1,1)+1;
        end
    end
    for i = 1 : ellc
        for j = 2 : n
            CM(i,j) = CM(i,j-1)+DeltaA/n;
        end
    end
    CM = mod(CM-1,ell)+1;
    
    for i = 1:ell
        C{i} = i:ell:DeltaA;
        D{i} = 0;
    end
    
    for ii = 1:n
        for jj = 1:ellc
            zz = zeros(1,DeltaA);
            yy = C{CM(jj,ii)};
            ee = func(D{CM(jj,ii)}+1:D{CM(jj,ii)}+denA,kA);
            D{CM(jj,ii)} = func(D{CM(jj,ii)}+denA, kA);
            zz(yy(ee))=1;
            rrr = random_matA{trial}((ii-1)*ellc+jj,:);
            Coding_matrixA{ii}(ellu+jj,:)= rrr.*zz;
        end
    end
    
    aa = [ones(1,denB) zeros(1,DeltaB-denB)];
    random_matB{trial} = randn(n,DeltaB);           % Random matrix for B
    for i = 1:n
        Coding_matrixB{i} = random_matB{trial}(i,:).*circshift(aa,func(i-1,DeltaB));
    end
    
    for i =1:n
        Coding_matrixAB{i} = kron(Coding_matrixA{i},Coding_matrixB{i});
    end
    
    
    condition_no = zeros(nchoosek(n,threshold),1);
    [u,~]= size(choices);
    for kk = 1:u
        wor = choices(kk,:);
        R = [];
        for i = 1:threshold
            R = [R ; Coding_matrixAB{wor(i)}];
        end
        condition_no(kk) = cond(R);
        rank_R(kk) = rank(R);
    end
    
    worst_condition_no(trial) = max(condition_no);
    pos(trial) = find(condition_no == worst_condition_no(trial));
    worst_choice_of_workers{trial} = choices(pos(trial),:);
end

worst_cond_no_over_trials = min(worst_condition_no);
pos = find(worst_condition_no == worst_cond_no_over_trials);

M1 = ['The worst case condition number is ', num2str(worst_cond_no_over_trials),'.'];
fprintf('\n'); disp(M1);
M2 = ['The worst case includes workers ', num2str(worst_choice_of_workers{pos}),'.'];
fprintf('\n'); disp(M2);
fprintf('\n');