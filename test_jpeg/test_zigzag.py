import numpy as np
#Generic Code
#2D to 1D
import numpy as np
def takestwoD(arr):
  arr = np.asarray(arr);
  m , n = arr.shape
  result = [0 for i in range(n*m)]
  result = np.asarray(result);
  
  
  count = 0;
  for i in range(0,n + m):
    if(i%2 == 0):
      x = 0;
      y = 0;
      if (i<m):
        x = i;
      else:
        x = m - 1;
      if (i<m):
        y = 0;
      else:
        y = i - m + 1;
      while (x >= 0 and y < n):
        result[count] = arr[x][y];
        count = count +1;
        x = x - 1;
        y = y + 1;
    else:
      x = 0;
      y = 0;
      if (i<n):
        x = 0;
      else:
        x = i - n + 1;
      if (i<n):
        y = i;
      else:
        y = n - 1;
      while (x < m and y >= 0):
        result[count] = arr[x][y];
        count = count +1;
        x = x + 1;
        y = y - 1;
  return np.asarray(result);


#Generic Code
#1D to 2D
def takesoneD(arr,rows,cols):
  m = rows;
  n = cols;
  result = [[0 for i in range(n)] for j in range(m)] 
  result = np.asarray(result)
  arr = np.asarray(arr);
  
  
  count = 0;
  for i in range(0,n + m):
    if(i%2 == 0):
      x = 0;
      y = 0;
      if (i<m):
        x = i;
      else:
        x = m - 1;
      if (i<m):
        y = 0;
      else:
        y = i - m + 1;
      while (x >= 0 and y < n):
        result[x][y] = arr[count];
        count = count +1;
        x = x - 1;
        y = y + 1;
    else:
      x = 0;
      y = 0;
      if (i<n):
        x = 0;
      else:
        x = i - n + 1;
      if (i<n):
        y = i;
      else:
        y = n - 1;
      while (x < m and y >= 0):
        result[x][y] = arr[count];
        count = count +1;
        x = x + 1;
        y = y - 1;
  return np.asarray(result);


arr=np.asarray([[1,2,3],[4,5,6],[7,8,9]])
a=np.asarray(takesTwoD(arr))
b=takesoneD(a,arr.shape[0],arr.shape[1])
print(a)
print(b)