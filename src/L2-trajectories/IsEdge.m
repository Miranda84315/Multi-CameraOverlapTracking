function fromEdge = IsEdge( xy_point )
    x = xy_point(1);
    y = xy_point(2);
    if (x<=100 || x >= 1720 || y<=100 || y >= 880)
        fromEdge = 1;
    else 
        fromEdge = 0;
end