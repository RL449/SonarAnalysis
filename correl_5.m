function [P,nlags] = correl_5(ts1,ts2,lags,offset)

for i = 0:lags
    
    ng = 1;
    sx = 2;
    sy = 3;
    sxx = 4;
    syy = 5;
    sxy = 6;
    
     for k = 1:(length(ts1)-(i+offset))
         x = ts1(k);
         y = ts2(k+(i+offset));
         
         if ~isnan(x) && ~isnan(y)
             
          sx = sx + x;                
          sy = sy + y;
          sxx = sxx + x*x;
          syy = syy + y*y;
          sxy = sxy + x * y;  
          ng = ng + 1;
          
         end
     end
     
     covar1 = (sxy/ng)-((sx/ng)*(sy/ng));
     denom1 = sqrt((sxx/ng)-(sx/ng)^2);
     denom2 = sqrt((syy/ng)-(sy/ng)^2);
     P(i+1) = covar1/(denom1*denom2);
    
     nlags = [0:lags];
     
end