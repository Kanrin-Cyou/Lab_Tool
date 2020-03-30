import numpy as np

def reshape(img,lim=20,th=5):

    img = np.array(img)
    index = np.argmax(img)
    
    x_index = img.shape[0]
    y_index = img.shape[1]

    x_centr = index//y_index
    y_centr = index%y_index
    
    n=0 
    i=0
    while n<lim: 
        if img[x_centr-i,y_centr]>th:
            i=i+1
        else:
            i=i+1
            n=n+1
            x_lim1=x_centr-i

    n=0 
    i=0
    while n<lim: #find 10 value<5 points
        if img[x_centr+i,y_centr]>th:
            i=i+1
        else:
            i=i+1
            n=n+1
            x_lim2=x_centr+i

    n=0 
    i=0
    while n<lim: 
        if img[x_centr,y_centr-i]>th:
            i=i+1
        else:
            i=i+1
            n=n+1
            y_lim1=y_centr-i        

    n=0 
    i=0
    while n<lim: 
        if img[x_centr,y_centr+i]>th:
            i=i+1
        else:
            i=i+1
            n=n+1
            y_lim2=y_centr+i

    limit=[x_lim1,x_lim2,y_lim1,y_lim2]        
    
    x_d=limit[1]-limit[0]
    y_d=limit[3]-limit[2]

    if x_d<y_d:
        c_d=y_d
    else:
        c_d=x_d

    c_d= c_d//2
    
    img=img[x_centr-c_d:x_centr+c_d,y_centr-c_d:y_centr+c_d]
    
    return img

def info(img):

    img = np.array(img)

    x_index = img.shape[0]
    y_index = img.shape[1]

    index=np.argmax(img)
    x= index//y_index
    y= index%y_index

    x_line=img[x,] 
    #- min(img[x,])
    y_line=img[:,y] 
    #- min(img[:,y])
    
    info = [x,x_line,y,y_line]  
    
    return info