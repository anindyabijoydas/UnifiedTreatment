%   Finding the worst case condition number for the polynomial code scheme
%   We have n workers and s = n - kA * kB stragglers.
%   Storage fraction gammaA = 1/kA and gammaB = 1/kB.
%   Set DeltaA = kA and DeltaB = kB.
%   We choose n nodes uniformly spaced in [-1,1].
%
%   This code uses the approach of the following paper-
%
%   Qian Yu, Mohammad Maddah-Ali, and Salman Avestimehr. Polynomial codes: 
%   an optimal design for highdimensional coded matrix multiplication. 
%   In Proc. of Advances in Neural Information Processing Systems
%   (NIPS), pages 4403–4413, 2017


clc
close all
clear

n = 24;
kA = 4;
kB = 5;
k = kA*kB;
s = n - k;

node_points = -1 + 2*(1:n)'/n;                  %% Choosing nodes in [-1,1]
Choice_of_workers = combnk(1:n,k);
[total_no_choices,~] = size(Choice_of_workers);
cond_no = zeros(total_no_choices,1);

for i=1:total_no_choices
    dd = Choice_of_workers(i,:);
    nodes = node_points(dd);
    Coding_matrix = zeros(k,k);
    for j=1:k
        Coding_matrix(j,:) = (nodes(j)).^((1:k)-1);
    end
    cond_no(i) = cond(Coding_matrix);
end

worst_condition_number = max(cond_no);
pos = find(cond_no == max(cond_no));
worst_choice_of_workers = Choice_of_workers(pos,:);

M1 = ['The worst case condition number is ', num2str(worst_condition_number),'.'];
fprintf('\n'); disp(M1);
M2 = ['The worst case includes workers ', num2str(worst_choice_of_workers),'.'];
fprintf('\n'); disp(M2);
fprintf('\n');
