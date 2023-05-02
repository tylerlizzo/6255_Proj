function descStats(x)

    min_ = min(x);
    max_ = max(x);
    median_ = median(x);
    mean_ = mean(x);
    std_ = std(x);

    fprintf("%-7s\t%-7s\t%-7s\t%-7s\t%-7s\n", "min", "median", "mean", "max", "std");
    fprintf("%-3.2f\t%-3.2f\t%-3.2f\t%-3.2f\t%-3.2f\n", min_, median_, mean_, max_, std_);
end