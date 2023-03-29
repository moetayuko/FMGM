function A = ConstructA_NP(TrainData, Anchor, k)
    anchor_num = size(Anchor, 2);
    num = size(TrainData, 2);
    A = sparse(num, anchor_num);

    %d*n
    dis = pdist2(TrainData', Anchor', 'squaredeuclidean');
    [dis, idx] = mink(dis, k + 1, 2);

    row = repmat(1:num, 1, k);
    col = reshape(idx(:, 1:k), 1, []);
    ind = sub2ind(size(A), row, col);

    A(ind) = (dis(:, k + 1) - dis(:, 1:k)) ./ (k * dis(:, k + 1) - sum(dis(:, 1:k), 2));
end
