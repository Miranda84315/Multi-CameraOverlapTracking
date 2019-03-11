function fromEdge = IsEdge3D( xy_point )
    x = xy_point(1);
    y = xy_point(2);
    if (x<=100 || x >= 1400 || y<=100 || y >= 1300)
        fromEdge = 1;
    else 
        fromEdge = 0;
end