function [X, y] = load_dataset(id)
    switch id
        case 1
            file = load('3sources');
            X = file.X;
            y = file.Y;
        case 2
            file = load('Caltech101-7');
            X = file.X;
            y = file.Y;
        case 3
            file = load('Caltech101-20');
            X = file.X;
            y = file.Y;
        case 4
            file = load('UCI');
            X = file.X;
            y = file.Y;
        case 5
            file = load('Caltech101-all');
            X = file.X;
            y = file.Y;
        case 6
            file = load('CIFAR10_llc_with_img_fea');
            X = cellfun(@transpose, file.X, 'uni', 0);
            X{1} = im2double(X{1});
            y = double(file.Y);
        case 7
            file = load('FashionMNIST_llc_with_img_fea');
            X = cellfun(@transpose, file.X, 'uni', 0);
            X{1} = im2double(X{1});
            y = double(file.Y);
        otherwise
            error('Unknown dataset')
    end
    X = cellfun(@(Xv) (Xv - mean(Xv, 2)) ./ std(Xv, 0, 2), X, 'uni', 0);
end
