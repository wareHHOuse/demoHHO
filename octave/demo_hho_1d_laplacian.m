%
% The code is an 1D implementation of the HHO method for the Laplace
% equation and illustrates the construction of the HHO operators and
% the assembly of the global problem, including static condensation.
%
% This code is in the public domain, you are free to do whatever you
% like with it. The program's only purpose is to demonstrate the HHO
% method; the authors take no responsability for any problem it could
% result from its usage.
%
% The main repository of this code is at
%
%                   https://github.com/wareHHOuse/demoHHO
%
% Please refer to the GIT log for the change history and author information.
% Have fun with HHO!
% 

function hho()
    clear;
    pd = initialize_hho();
    
    % Uncomment this to showcase the reduction operator
    %test_reduction(pd);

    % Uncomment this to showcase the reconstruction operator
    %test_reconstruction(pd);

    % Uncomment this to convergence-test the stabilization operator
    %test_stabilization(pd);

    % Uncomment this to convergence-test HHO
    test_hho(pd);
end

function pd = initialize_hho()
    pd = struct;
    pd.N = 4;       % number of elements
    pd.K = 0;       % polynomial degree
    pd.h = 1/pd.N;
    pd.tp = 16;     % plot points
end

% Problem right-hand side
function rhs = rhs_fun(x)
    rhs = sin(pi*x) * pi^2;
end

% Problem analytical solution
function sol = sol_fun(x)
    sol = sin(pi*x);
end

% Evaluate scalar monomial basis
function [phi, dphi] = basis(x, x_bar, h, max_k)
    k       = (0:max_k)';
    x_tilde = 2*(x-x_bar)/h;
    
    phi = x_tilde .^ k;
    
    dphi    = zeros(max_k+1,1);
    dphi(2:end) = (2*k(2:end)/h).*(x_tilde.^k(1:end-1));
end

% Make mass matrix for the specified element
function MM = make_mass_matrix(pd, elem, order)
    x_bar = cell_center(pd, elem);
    [qps, qws, nn] = integrate(2*order, pd.h, elem);
    MM = zeros(order+1, order+1);
    for ii = 1:nn
        [phi, ~] = basis(qps(ii), x_bar, pd.h, order);
        MM = MM + qws(ii) * (phi * phi');
    end
end

% Compute RHS term for the specified element
function rhs = make_rhs(pd, elem, order, fun)
    x_bar = cell_center(pd, elem);
    [qps, qws, nn] = integrate(2*order+1, pd.h, elem);
    rhs = zeros(order+1, 1);
    for ii = 1:nn
        [phi, ~] = basis(qps(ii), x_bar, pd.h, order);
        rhs = rhs + qws(ii) * phi * fun(qps(ii));
    end
end

% The HHO reduction operator
function I = hho_reduction(pd, elem, fun)
    x_bar = cell_center(pd, elem);
    [xl, xr] = face_centers(pd, elem);
    [qps, qws, nn] = integrate(2*pd.K, pd.h, elem);
    MM = zeros(pd.K+1, pd.K+1);
    rhs = zeros(pd.K+1, 1);
    for ii = 1:nn
        [phi, ~] = basis(qps(ii), x_bar, pd.h, pd.K);
        MM = MM + qws(ii) * (phi * phi');           % Mass matrix
        rhs = rhs + qws(ii) * phi * fun(qps(ii));   % Right-hand side
    end
    I = zeros(pd.K+3, 1);
    I(1:pd.K+1) = MM\rhs;   % Project on the cell
    I(pd.K+2) = fun(xl);    % Project on faces: in 1D we just need
    I(pd.K+3) = fun(xr);    %   to evaluate function at the faces
end

function test_reduction(pd)
    f = @(x) sin(pi*x);
    
    hold on;
    
    error = 0.0;
    for ii = 1:pd.N
        x_bar = cell_center(pd, ii);
        Iv = hho_reduction(pd, ii, f);
        
        [qps, qws, nn] = integrate(2*(pd.K+1), pd.h, ii);
        for jj = 1:nn
            [phi, ~] = basis(qps(jj), x_bar, pd.h, pd.K);
            error = error + qws(jj) * ( dot(Iv(1:end-2), phi) - f(qps(jj)) )^2;
        end
        
        tps = make_test_points(pd, ii);
        fs = arrayfun(f,tps);
        
        plot(tps, fs, 'b');
        
        pfx = zeros(pd.tp,1);
        for jj = 1:pd.tp
            [phi, ~] = basis(tps(jj), x_bar, pd.h, pd.K);
            pfx(jj) = dot(Iv(1:end-2), phi);
        end
  
        plot(tps, pfx, 'r');      
        [xl, xr] = face_centers(pd, ii);
        plot([xl, xr], [Iv(end-1),Iv(end)], '*k');
    end
    
    hold off; 

    disp( sprintf('Projection error: %g', sqrt(error)) );
end

% The HHO reconstruction operator
function [A, R] = hho_reconstruction(pd, elem)
    x_bar = cell_center(pd, elem);
    [xF1, xF2] = face_centers(pd, elem);

    stiff_mat = zeros(pd.K+2, pd.K+2);
    gr_rhs = zeros(pd.K+1, pd.K+3);
    
    [qps, qws, nn] = integrate(2*(pd.K+1), pd.h, elem);
    for ii = 1:nn
        [~, dphi] = basis(qps(ii), x_bar, pd.h, pd.K+1);
        stiff_mat = stiff_mat + qws(ii) * (dphi * dphi');
    end
    
    % Set up local Neumann problem
    gr_lhs = stiff_mat(2:end, 2:end); % Left hand side
    % Right hand side, cell part
    gr_rhs(:,1:pd.K+1) = stiff_mat(2:end,1:pd.K+1); % (grad uT, grad v)
    
    [phiF1, dphiF1] = basis(xF1, x_bar, pd.h, pd.K+1);
    [phiF2, dphiF2] = basis(xF2, x_bar, pd.h, pd.K+1);
    
    % Right hand side, face part
    gr_rhs(1:end, 1:pd.K+1) = gr_rhs(1:end, 1:pd.K+1) + ...
        dphiF1(2:end)*phiF1(1:pd.K+1)'; % (uT, n.grad v)
    
    gr_rhs(1:end, 1:pd.K+1) = gr_rhs(1:end, 1:pd.K+1) - ...
        dphiF2(2:end)*phiF2(1:pd.K+1)'; % (uT, n.grad v)
    
    gr_rhs(1:end, pd.K+2) = - dphiF1(2:end); % (uF, n.grad v)
    gr_rhs(1:end, pd.K+3) = + dphiF2(2:end); % (uF, n.grad v)
    
    R = gr_lhs\gr_rhs;   % Solve problem 
    A = gr_rhs'*R;       % Compute (grad(Ru), grad(Rv))
end


function test_reconstruction(pd)
    f = @(x) sin(pi*x);
    
    x   = [];
    fx  = [];
    rfx = [];
    
    error = 0.0;
    for ii = 1:pd.N
        x_bar = cell_center(pd, ii);
        
        Iv = hho_reduction(pd, ii, f);
        [~,oper] = hho_reconstruction(pd, ii);
        v_star = oper * Iv;
        v = [Iv(1); v_star];
        
        [qps, qws, nn] = integrate(2*(pd.K+1), pd.h, ii);
        for jj = 1:nn
            [phi, ~] = basis(qps(jj), x_bar, pd.h, pd.K+1);
            error = error + qws(jj) * ( dot(v, phi) - f(qps(jj)) )^2;
        end
        
        tps = make_test_points(pd, ii);
        for jj = 1:length(tps)
            x = [x;tps(jj)];
            fx = [fx; f(tps(jj))];
            [phi, ~] = basis(tps(jj), x_bar, pd.h, pd.K+1);
            rfx = [rfx; dot(v, phi)];
        end
    end
    plot(x, fx);
    hold on;
    plot(x, rfx);
    
    disp( sprintf('Reconstruction error: %g', sqrt(error)) );
end

function S = hho_stabilization(pd, elem, R)
    xT = cell_center(pd, elem);
    [xF1, xF2] = face_centers(pd, elem);

    mm = make_mass_matrix(pd, elem, pd.K+1);
    
    % compute the term tmp1 = uT - ??T(R(u))
    M = mm(1:pd.K+1,1:pd.K+1);
    Q = mm(1:pd.K+1,2:pd.K+2);
    tmp1 = - M\(Q*R);
    tmp1(1:pd.K+1, 1:pd.K+1) = tmp1(1:pd.K+1, 1:pd.K+1) + eye(pd.K+1);
    
    [phiF1, ~] = basis(xF1, xT, pd.h, pd.K+1);
    Mi = 1;
    Ti = phiF1(2:end)';
    Ti_hat = phiF1(1:pd.K+1)';
    tmp2 = Mi \ (Ti*R);             % tmp2 = ??F(R(u))
    tmp2(pd.K+2) = tmp2(pd.K+2)-1;  % tmp2 = uF - ??F(R(u))
    tmp3 = Mi \ (Ti_hat * tmp1);    % tmp3 = ??F(uT - ??T(R(u)))
    Si = tmp2 + tmp3;               % Si = uF - ??F(R(u)) + ??F(uT - ??T(R(u)))
    S = Si' * Mi * Si / pd.h;       % Accumulate on S
    
    [phiF2, ~] = basis(xF2, xT, pd.h, pd.K+1);
    Mi = 1;
    Ti = phiF2(2:end)';
    Ti_hat = phiF2(1:pd.K+1)';
    tmp2 = Mi \ (Ti*R);             % tmp2 = ??F(R(u))
    tmp2(pd.K+3) = tmp2(pd.K+3)-1;  % tmp2 = uF - ??F(R(u))
    tmp3 = Mi \ (Ti_hat * tmp1);    % tmp3 = ??F(uT - ??T(R(u)))
    Si = tmp2 + tmp3;               % Si = uF - ??F(R(u)) + ??F(uT - ??T(R(u)))
    S = S + Si' * Mi * Si / pd.h;   % Accumulate on S
end

function l2err = run_stabilization(pd)    
    error = 0;
    for ii = 1:pd.N
        x_bar = cell_center(pd, ii);
        [A,R] = hho_reconstruction(pd, ii);
        S = hho_stabilization(pd, ii, R);

        MM = make_mass_matrix(pd, ii, pd.K);
    
        sf = @(x) sol_fun(x);
        sf_red = hho_reduction(pd, ii, sf);

        error = error + sf_red'*S*sf_red;
    end

    l2err = sqrt(error);
end

function test_stabilization(pd)
    orders = [0,1,2,3,4];
    elems = [2,4,8,16,32];

    for K = orders
        err = [];
        for N = elems
            pd.K = K;
            pd.N = N;
            pd.h = 1/N;
            e = run_stabilization(pd);
            err = [err,e];
        end

        hold on;
        loglog(elems,err);
        drawnow;
        hold off;
        end
end


function [LC, rhsC] = static_condensation(L, rhs)
    LTT = L(1:end-2, 1:end-2);
    LTF = L(1:end-2, end-1:end);
    LFT = L(end-1:end, 1:end-2);
    LFF = L(end-1:end, end-1:end);
    LC = LFT*(LTT\LTF) - LFF;
    rhsC = LFT*(LTT\rhs);
end

function uT = static_decondensation(L, rhs, uF)
    LTT = L(1:end-2, 1:end-2);
    LTF = L(1:end-2, end-1:end);
    uT = LTT\(rhs - LTF*uF);
end

function l2err = run_hho(pd)
    
    gLHS = sparse(pd.N+1, pd.N+1); % Zero Dirichlet at both sides
    gRHS = zeros(pd.N+1, 1);

    % Assemble global system
    for ii = 1:pd.N
        x_bar = cell_center(pd, ii);
        [A,R] = hho_reconstruction(pd, ii);
        S = hho_stabilization(pd, ii, R);
        L = A + S;
        
        [qps, qws, nn] = integrate(2*(pd.K+1), pd.h, ii);
        rhs = zeros(pd.K+1,1);
        for jj = 1:nn
            [phi, ~] = basis(qps(jj), x_bar, pd.h, pd.K);
            rhs = rhs + qws(jj) * rhs_fun(qps(jj)) * phi;
        end

        [LC, rhsC] = static_condensation(L, rhs);

        for rr = 0:1
            for cc = 0:1
                gLHS(ii+rr, ii+cc) = gLHS(ii+rr, ii+cc) + LC(rr+1, cc+1);
            end
            gRHS(ii+rr) = gRHS(ii+rr) + rhsC(rr+1);
        end
    end
    
    % Remove rows/cols corresponding to initial
    % and final node (zero Dirichlet)
    gLHS(1,:) = [];
    gLHS(:,1) = [];
    gLHS(end,:) = [];
    gLHS(:,end) = [];
    gRHS = gRHS(2:end-1);

    solution = gLHS\gRHS;

    % Postprocess
    error = 0.0;
    for ii = 1:pd.N
        x_bar = cell_center(pd, ii);
        [A,R] = hho_reconstruction(pd, ii);
        S = hho_stabilization(pd, ii, R);
        L = A + S;
        
        [qps, qws, nn] = integrate(2*(pd.K+1), pd.h, ii);
        rhs = zeros(pd.K+1,1);
        for jj = 1:nn
            [phi, ~] = basis(qps(jj), x_bar, pd.h, pd.K);
            rhs = rhs + qws(jj) * rhs_fun(qps(jj)) * phi;
        end

        lsol_F = [];
        if ii == 1
            lsol_F = [0; solution(1)];
            else if ii == pd.N
                lsol_F = [solution(pd.N-1); 0];
            else
                lsol_F = [solution(ii-1); solution(ii)];
            end
        end

        lsol_T = static_decondensation(L, rhs, lsol_F);

        MM = make_mass_matrix(pd, ii, pd.K);
    
        sf = @(x) sol_fun(x);
        refrhs = make_rhs(pd, ii, pd.K, sf);
        refsol = MM \ refrhs;

        error = error + (refsol-lsol_T)' * (MM*(refsol-lsol_T));
    end
    l2err = sqrt(error);
end

function test_hho(pd)
    orders = [0,1,2,3,4,5];
    elems = [2,4,8,16,32,64];

    for K = orders
        err = [];
        for N = elems
            pd.K = K;
            pd.N = N;
            pd.h = 1/N;
            e = run_hho(pd);
            err = [err,e];
        end

        hold on;
        loglog(elems,err);
        drawnow;
        hold off;
    end
end

% Utility functions
function xT = cell_center(pd, ii)
    xT = (ii-0.5)*pd.h;
end

function [xl, xr] = face_centers(pd, ii)
    xl = (ii-1)*pd.h;
    xr = ii * pd.h;
end

function tp = make_test_points(pd, elem)
    [xl, xr] = face_centers(pd, elem);
    tp = linspace(xl, xr, pd.tp);
end

function [nodes, weights, num_nodes] = integrate(order, h, elem)
    if rem(order, 2) == 0
        order = order+1;
    end
    
    if rem(order,2) == 0
        num_nodes = floor((order+1)/2);
    else
        num_nodes = floor((order+2)/2);
    end
    num_nodes = max(num_nodes, 1);

    if num_nodes == 1
        nodes = h*(elem-0.5);
        weights = h;
        return;
    end
    M = zeros(num_nodes, num_nodes);
    for ii = 1:num_nodes-1
        v = sqrt(1/(4-1/((ii)^2)));
        M(ii+1,ii) = v;
        M(ii,ii+1) = v;
    end

    [d,v] = eig(M);
    tnodes = diag(v)';
    tweights = (d(1,:).^2);
    
    weights = tweights*h;
    all_ones = ones(size(tnodes));
    nodes = h*( (0.5*tnodes) + ((elem-0.5)*all_ones));
end

