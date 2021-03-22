function acc_valid_all11 = filter_win(acc_valid_all1,w)
if size(acc_valid_all1,2)==1
    if mod(length(w),2)==0
        l = length(w)/2;
        w = [w(1:l); w(l+2:end)];
    else
        l = (length(w)+1)/2;
    end
    acc_valid_all11 = conv(acc_valid_all1,w,'full');
    ll = length(w);
    nn = [1:ll ones(1,length(acc_valid_all1)-ll-1)*ll ll:-1:1]';
    acc_valid_all11 = acc_valid_all11./nn;
    acc_valid_all11 = acc_valid_all11(l:l+length(acc_valid_all1)-1);
else
    if mod(size(w,1),2)==0
        l = size(w,1)/2;
        w = [w(1:l,1:l) w(1:l,l+2:end); w(l+2:end,1:l) w(l+2:end,l+2:end)];
    else
        l = (size(w,1)+1)/2;
    end
    acc_valid_all11 = conv2(acc_valid_all1,w,'full');
    ll = size(w,1);
    nn = [1:ll ones(1,size(acc_valid_all1,1)-ll-1)*ll ll:-1:1]'; nn = nn*nn';
    acc_valid_all11 = acc_valid_all11./nn;
    acc_valid_all11 = acc_valid_all11(l:l+size(acc_valid_all1,1)-1,l:l+size(acc_valid_all1,1)-1);
end
