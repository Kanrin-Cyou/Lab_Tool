import numpy as np

def plot_sum(x,y):

    c=[]
    for i in range(len(x)):
        c.append([x[i],y[i]])
    c.sort()
    
    a=[]
    b=[] 
    for i in range(len(x)):
        a.append(c[i][0])
        b.append(c[i][1])
    x=a
    y=b
    
    sum=0
    for i in range(len(y)-1):
        deltax = x[i+1] - x[i]
        deltay = abs(y[i+1] - y[i])
        
        if y[i+1]>y[i]:
            rec = y[i]*deltax
        else:
            rec = y[i+1]*deltax
        
        tri = deltax*deltay/2
        
        sum = sum + rec + tri

    return sum

def index(y_line):
    
    val_width=(max(y_line)-min(y_line))*0.135
    
    y_line1=y_line[0:int(len(y_line)/2)]
    y_line2=y_line[int(len(y_line)/2):len(y_line)]
    
    index1 = np.argmin(np.abs(np.flip(np.array(y_line1))-min(y_line)-val_width))
    index2 = np.argmin(np.abs(np.array(y_line2)-min(y_line)-val_width))
    
    index1 = int(len(y_line)/2)-index1-1
    index2 = int(len(y_line)/2)+index2
    
    return index1,index2

def coverage(y_plot,y_line):
    
    index1,index2=index(y_line)
    y_line_width = y_line[index1:index2]
    
    y_plot=y_plot[index1:index2]
    
    sum_y_line_width=plot_sum(y_plot*5.2/1000,(y_line_width-min(y_line))/255*100)
    
    return sum_y_line_width

def coverage2(y_y,y_line,fit_line):
    
    index1,index2=index(fit_line)
    
    scale=len(y_line)/len(fit_line)
    
    index1=int(index1*scale)
    index2=int(index2*scale)

    y_line_width = y_line[index1:index2]
    y_y = y_y[index1:index2]
    sum_y_line_width=plot_sum(y_y*5.2/1000,(y_line_width-min(y_line))/255*100)
    
    return sum_y_line_width