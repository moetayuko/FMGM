function [y_ind, alpha, obj] = fmgm(B, Y)
%   J. Lu, F. Nie, R. Wang and X. Li, "Fast Multiview Clustering by Optimal
%   Graph Mining," IEEE Transactions on Neural Networks and Learning Systems,
%   doi: 10.1109/TNNLS.2023.3256066.
%
%   SPDX-FileCopyrightText: 2023 Jitao Lu <dianlujitao@gmail.com>
%   SPDX-License-Identifier: MIT
    num_views = size(B, 3);
    alpha = zeros(num_views, 1) + 1 / num_views;

    % cached variables
    H = calc_H(B);
    YB = pagemtimes(Y', B);
    yBBy = transmul(YB);

    y1 = sum(Y)';
    y_ind = vec2ind(Y')';

    for iter = 1:10
        S = update_S(yBBy, y1, alpha);
        [y_ind, ~, YB, yBBy, y1] = update_Y(B, S, y_ind, y1, alpha, YB);
        obj(iter) = calc_fmgm_obj(S, y1, alpha, H, yBBy);
        if iter > 2 && abs(obj(iter) - obj(iter - 1)) < 1e-5
            break;
        end
        alpha = update_alpha(H, yBBy, S);
    end
end

function obj = calc_fmgm_obj(S, y1, alpha, H, yBBy)
    obj = alpha' * H * alpha + (S' .^ 2) * (y1 .^ 2) - 2 * alpha' * (yBBy' * S);
end

function XX = transmul(X)
    XX = permute(sum(X .^ 2, 2), [1, 3, 2]);
end

function H = calc_H(B)
    num_views = size(B, 3);
    [idx_v, idx_w] = meshgrid(1:num_views);
    mask = idx_v <= idx_w;
    idx_v = idx_v(mask);
    idx_w = idx_w(mask);

    BB = pagemtimes(B(:, :, idx_v), 'transpose', B(:, :, idx_w), 'none');
    H = zeros(num_views);
    H(mask) = sum(BB .^ 2, [1, 2]);
    H = tril(H, -1)' + H;
end

function S = update_S(yBBy, y1, alpha)
    S = yBBy * alpha ./ full(y1 .^ 2);
end

function [y_ind, obj, YB, yBBy, y1] = update_Y(B, S, y_ind, y1, alpha, YB)
    n = size(y_ind, 1);

    skk_alpha_bb = 2 * S * (transmul(B) * alpha)';

    for iter = 1:15
        yBBy = transmul(YB);
        obj(iter) = (S' .^ 2) * (y1 .^ 2) - 2 * S' * (yBBy * alpha);
        if iter > 2 && abs(obj(iter - 1) - obj(iter)) < 1e-5
            break;
        end

        for ii = 1:n
            p = y_ind(ii);
            % avoid generating empty cluster
            if y1(p) == 1
                continue;
            end

            bi = B(ii, :, :);
            YBbi = pagemtimes(YB, 'none', bi, 'transpose');
            YBbi = permute(YBbi, [1, 3, 2]);

            delta = S .^ 2 .* (2 * y1 + 1) ...
                    - 4 * S .* (YBbi * alpha) - skk_alpha_bb(:, ii);
            delta(p) = delta(p) - 2 * (S(p) ^ 2) + 2 * skk_alpha_bb(p, ii);

            [~, q] = min(delta);
            if q ~= p
                y1([p, q]) = y1([p, q]) + [-1; 1];
                y_ind(ii) = q;

                YB(p, :, :) = YB(p, :, :) - bi;
                YB(q, :, :) = YB(q, :, :) + bi;
            end
        end
    end
end

function alpha = update_alpha(H, yBBy, S)
    f = yBBy' * S;

    num_views = size(H, 1);
    A = [];
    b = [];
    Aeq = ones(1, num_views);
    beq = 1;
    lb = zeros(num_views, 1);
    ub = [];
    x0 = [];
    alpha = quadprog(H, -f, A, b, Aeq, beq, lb, ub, x0, ...
            optimset('Display', 'off'));
end

% vim: tw=79 ts=4 sw=4
