import pygame
import cv2
import numpy as np
import requests

threshold=60
close_neighbors=8

pygame.init()
width,height=500,500
screen=pygame.display.set_mode((width,height))
pygame.display.set_caption("Choose points for pathfinding")
response = requests.get("https://raw.githubusercontent.com/akshatkkaushik/ARK-Perception-Task/refs/heads/main/maze.png", stream=True).raw
#image_array = np.asarray(bytearray(response.read()), dtype=np.uint8)

image_data = response.content
image_file = io.BytesIO(image_data)
maze=pygame.image.load(image_file)
maze=pygame.transform.scale(maze, (width, height))
start_py= None
end_py= None
running= True

while running:
    screen.blit(maze, (0, 0))
    if start_py:
        pygame.draw.circle(screen,(128,0,128),start_py,5)

    for event in pygame.event.get():
        if event.type== pygame.QUIT:
            running= False
        elif event.type==pygame.MOUSEBUTTONDOWN:
            if event.button== 1:  # Left mouse button
                if start_py is None:
                    start_py = event.pos
                elif end_py is None:
                    end_py = event.pos
            if start_py and end_py:
                pygame.draw.circle(screen,(127,0,255),end_py,5)
                running = False
    pygame.display.update()
pygame.quit()

if start_py and end_py:
    start =[start_py[1], start_py[0]]  
    end =[end_py[1], end_py[0]]

    def checkValid(x_1,y_1,x_2,y_2,image):
        mask = np.zeros_like(image, dtype=np.uint8)
        cv2.line(mask,(y_1,x_1),(y_2,x_2),255,1)
        linepixels=image[mask==255]
        return np.all(linepixels==255)
    def closest(node_list, current):
        distance_list=[]
        ind_list=[]
        for i in range(len(node_list)):
            x=node_list[i][0]
            y=node_list[i][1]
            dis=np.sqrt((x-current[0])**2+(y-current[1])**2)
            if(dis<threshold):
                distance_list.append(dis)
                ind_list.append(i)
        sorted_dist=sorted(distance_list)
        final_dist=[]
        final_ind=[]
        for i in range(close_neighbors if len(sorted_dist)>close_neighbors else len(sorted_dist)):
            final_dist.append(sorted_dist[i])
            final_ind.append(ind_list[distance_list.index(sorted_dist[i])])
        return final_ind,final_dist
    # start_hard=[30,165]
    # end_hard=[425,465]
    # start_easy=[455,50]
    # end_easy=[455,100]
    # start=[30,165]
    # end=[425,465]
    x0=start[0]
    y0=start[1]
    nodes=[start]
    d=[0]
    child_nodes=[[]]
    cost=[[]]
    response = requests.get("https://raw.githubusercontent.com/akshatkkaushik/ARK-Perception-Task/refs/heads/main/maze.png", stream=True).raw
    image_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
    img=cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
    for i in range(0,500):
        for j in range(0,500):
            if img[i][j]>=180:
                img[i][j]=255
            else:
                img[i][j]=0
    img2=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    cv2.ellipse(img2,(end[1],end[0]),(3,3),0,0,360,(0,0,0),-1)
    cv2.ellipse(img2,(start[1],start[0]),(3,3),0,0,360,(0,0,0),-1)
    k=1
    while k<1000:
        flag=0
        #gnerating random points considering the edges of the maze
        x=np.random.randint(30,450)
        y=np.random.randint(20,470)
        if(img[x,y]!=0):
            idx_list,dis_list=closest(nodes,[x,y])
            for op,idx in enumerate(idx_list):
                if(checkValid(x,y,nodes[idx][0],nodes[idx][1],img)):
                    flag+=1
                    if(flag==1):
                        nodes.append([x,y])
                        k+=1
                        d.append(float('inf'))
                        new_idx=nodes.index([x,y])
                        child_nodes.append([])
                        cost.append([])
                    cv2.ellipse(img2,(y,x),(2,2),0,0,360,(0,0,255),-1)
                    child_nodes[idx].append(new_idx)
                    cost[idx].append(dis_list[op])
                    child_nodes[new_idx].append(idx)
                    cost[new_idx].append(dis_list[op])
                    cv2.line(img2,(y,x),(nodes[idx][1],nodes[idx][0]),(0,255,0),1)
                    cv2.imshow("HEllo",img2)
                    cv2.waitKey(1)
                
    visited=[]
    min_ind=0
    while(True):
        visited.append(nodes[min_ind])
        for j,kk in enumerate(child_nodes[min_ind]):
            if(d[min_ind]+cost[min_ind][j]<d[kk]):
                d[kk]=d[min_ind]+cost[min_ind][j]
        min_distance = float('inf')
        for l in range(len(nodes)):
            if d[l] < min_distance and nodes[l] not in visited:
                min_distance = d[l]
                min_ind = l
        if min_distance == float('inf'):
            break 
    print(d)
    last_dis_min=float('inf')
    d_min= float('inf')
    min_ind=-1
    end_close,_=closest(nodes,end)
    for i in end_close:
        if d[i]<d_min and checkValid(end[0],end[1],nodes[i][0],nodes[i][1],img):
            d_min=d[i]
            min_ind=i
    curr_ind=min_ind
    if curr_ind!=-1:
        current=nodes[curr_ind]
        print(len(nodes))
        cv2.line(img2,(current[1],current[0]),(end[1],end[0]),(0,0,255),3)
        cv2.waitKey(500)
        while current!=start:
            dmin=float('inf')
            childmin_ind = curr_ind 
            for i in child_nodes[curr_ind]:
                if d[i]<dmin :
                    dmin=d[i]
                    childmin_ind=i
            cv2.line(img2,(current[1],current[0]),(nodes[childmin_ind][1],nodes[childmin_ind][0]),(0,0,255),3)
            curr_ind=childmin_ind
            current=nodes[curr_ind]
            cv2.imshow("HEllo",img2)
            cv2.waitKey(500)

        cv2.ellipse(img2,(end[1],end[0]),(3,3),0,0,360,(128,128,0),-1)
        cv2.ellipse(img2,(start[1],start[0]),(3,3),0,0,360,(128,128,0),-1)
        cv2.imshow("HEllo",img2)
        cv2.waitKey(0)
    else:
        print("No path found / Insufficient nodes to find a path, try increasing the number of nodes")
else:
    print("point not selected")
