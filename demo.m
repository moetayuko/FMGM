addpath('funs');
addpath('data');

[X, y] = load_dataset(1);

n = size(X{1}, 1);
k = 5;
num_clusters = numel(unique(y));
numanchors = round(sqrt(n * k));
numanchors = 2 ^ round(log2(numanchors));

Y_init = full(ind2vec(kmeans(X{1}, num_clusters)')');

%%
Z = cellfun(@(Xv) build_bipartite_graph(Xv, k, numanchors), X, 'uni', 0);
B = cellfun(@(Zv) full(Zv ./ sqrt(sum(Zv))), Z, 'uni', 0);
B = cat(3, B{:});

%%
tic;
[y_pred, coeff, obj] = fmgm(B, Y_init);
time_elapsed = toc;

result = ClusteringMeasure_new(y, y_pred);

fprintf('time=%f\n', time_elapsed);
disp(result);

function Z = build_bipartite_graph(X, k, numanchors)
    n = size(X, 1);
    assert(numanchors < n);

    hn = log2(numanchors);
    [~, anchors] = hKM(X', 1:n, hn, 1);
    Z = ConstructA_NP(X', anchors, k);
end
