function BB = column_kron(Bmats)

switch length(Bmats)
    case 1
        BB = Bmats{1};
    case 2
        B1 = permute(Bmats{1},[2,1]);
        B2 = permute(Bmats{2},[2,1]);
        n1 = size(B1,2);
        n2 = size(B2,2);
        d = size(B1,1);
        B11 = repmat(B1,[1,1,n2]);
        B22 = permute(repmat(B2,[1,1,n1]),[1,3,2]);
        BB = B11.*B22;
        BB = reshape(BB,d,[])';
    case 3
        B1 = permute(Bmats{1},[2,1]);
        B2 = permute(Bmats{2},[2,1]);
        n1 = size(B1,2);
        n2 = size(B2,2);
        d = size(B1,1);
        B11 = repmat(B1,[1,1,n2]);
        B22 = permute(repmat(B2,[1,1,n1]),[1,3,2]);
        BB = B11.*B22;
        B12 = reshape(BB,d,[]);
        
        
        B3 = permute(Bmats{3},[2,1]);
        n12 = size(B12,2);
        n3 = size(B3,2);
        B1212 = repmat(B12,[1,1,n3]);
        B33 = permute(repmat(B3,[1,1,n12]),[1,3,2]);
        BB = B1212.*B33;
        BB = reshape(BB,d,[])';
end