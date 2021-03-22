function xx = or_op(x)
xx = false(1,size(x,2));
for i=1:size(x,1)
    xx = xx | logical(x(i,:));
end
xx = double(xx);