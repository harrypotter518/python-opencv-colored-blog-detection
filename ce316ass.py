import cv2  
import numpy as np
import math
import sys

def distance(x1, y1,x2,y2):
	return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))

def distance3d(x1, y1,z1,x2,y2,z2):
	return math.sqrt(((x1-x2)/pixel_size)*((x1-x2)/pixel_size)+((y1-y2)/pixel_size)*((y1-y2)/pixel_size)+(z1-z2)*(z1-z2))

colors=['blue','green','red','yellow','gold','gray','tblue']
lowerLimit =[]
color_value=[]
upperLimit = []
hsv_val=[]
dist=[0,0,0,0,0,0,0]
last_x_pos=[0,0,0,0,0,0,0]
last_y_pos=[0,0,0,0,0,0,0]
last_z_pos=[0,0,0,0,0,0,0]
last_x_pos2=[0,0,0,0,0,0,0]
last_y_pos2=[0,0,0,0,0,0,0]
last_di_cos_x=[0,0,0,0,0,0,0]
last_di_cos_y=[0,0,0,0,0,0,0]
last_di_cos_z=[0,0,0,0,0,0,0]
last_di_cos_x2=[0,0,0,0,0,0,0]
last_di_cos_y2=[0,0,0,0,0,0,0]
last_di_cos_z2=[0,0,0,0,0,0,0]
ufo=[0,0,0,0,0,0,0]
pixel_size= float(10/1000000)
f_length=12
baseline=3500
disparity=0
title=""
uf_str="UFO: "

for i in range(len(colors)):
	if i==0:
		color_value.append(np.uint8([[[255, 0, 0]]]))
	elif i==1:
		color_value.append(np.uint8([[[0, 255, 0]]]))
	elif i==2:
		color_value.append(np.uint8([[[0, 0, 200]]]))
	elif i==3:
		color_value.append(np.uint8([[[0, 180, 180]]]))
	elif i==4:
		color_value.append(np.uint8([[[0, 120, 200]]]))
	elif i==5:
		color_value.append(np.uint8([[[200, 200, 200]]]))
	else:
		color_value.append(np.uint8([[[150, 150, 0]]]))

	hsv_val.append(cv2.cvtColor(color_value[i], cv2.COLOR_BGR2HSV))

	if i == 0:
		low=hsv_val[i][0][0][0] - 15, 100, 100	
		up=hsv_val[i][0][0][0] + 15, 255, 255
	elif i == 3:
		low=hsv_val[i][0][0][0] - 5, 100, 100	
		up=hsv_val[i][0][0][0] + 10, 255, 255
	else:
		low=hsv_val[i][0][0][0] - 10, 100, 100	
		up=hsv_val[i][0][0][0] + 10, 255, 255
	
	lowerLimit.append(low)
	upperLimit.append(up)

f = open("dist.txt", "w")

nframes = int(sys.argv[1])

for frame in range(nframes):

	fn_left = sys.argv[2]% frame
	fn_right = sys.argv[3]% frame

	image = cv2.imread(fn_left)
	image2 = cv2.imread(fn_right)

	ufo1_id=0
	ufo2_id=0	
	cnt=[0,0,0,0,0,0,0]
	x_pos=[0,0,0,0,0,0,0]
	y_pos=[0,0,0,0,0,0,0]	
	z_pos=[0,0,0,0,0,0,0]	
	cnt2=[0,0,0,0,0,0,0]
	x_pos2=[0,0,0,0,0,0,0]
	y_pos2=[0,0,0,0,0,0,0]	
	di_cos_x=[0,0,0,0,0,0,0]
	di_cos_y=[0,0,0,0,0,0,0]
	di_cos_z=[0,0,0,0,0,0,0]	
	di_cos_x2=[0,0,0,0,0,0,0]
	di_cos_y2=[0,0,0,0,0,0,0]
	di_cos_z2=[0,0,0,0,0,0,0]
	mask=[]
	mask2=[]
	
	for i in range(len(colors)):
		hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #convert the image into hsv
		hsv2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV) #convert the image into hsv
		lv = np.array(lowerLimit[i]) 
		uv = np.array(upperLimit[i])

		mask.append(cv2.inRange(hsv, lv, uv))
		mask2.append(cv2.inRange(hsv2, lv, uv))	

	for y1 in range(0,480):
		for x1 in range(0,640):
			for i in range(len(colors)):
				if i==5:
					b,g,r = image[y1,x1]
					b = int(b)
					g = int(g)
					r = int(r)
					b2,g2,r2 = image2[y1,x1]
					b2 = int(b2)
					g2 = int(g2)
					r2 = int(r2)

					if abs(b-r)<20 and abs(b-g)<20 and g>100:
						cnt[i] = cnt[5]+1
						x_pos[i]+=x1
						y_pos[i]+=y1
					if abs(b2-r2)<20 and abs(b2-g2)<20 and g2>100:
						cnt2[i] = cnt2[5]+1
						x_pos2[i]+=x1
						y_pos2[i]+=y1
				else:
					if mask[i][y1,x1]>0:
						cnt[i] = cnt[i]+1
						x_pos[i]+=x1
						y_pos[i]+=y1
					if mask2[i][y1,x1]>0:
						cnt2[i] = cnt2[i]+1
						x_pos2[i]+=x1
						y_pos2[i]+=y1		

	if frame>0:	
		title="frame	identity	distance"
	print (title)
	f.writelines(title+"\n")
	
	for i in range(len(colors)):		
	
		if cnt[i]>0:
			x_pos[i] =x_pos[i]/cnt[i]
			y_pos[i] =y_pos[i]/cnt[i]
		else:
			x_pos[i]= last_x_pos[i]
			y_pos[i]=last_y_pos[i]

		if cnt2[i]>0:
			x_pos2[i] =x_pos2[i]/cnt2[i]
			y_pos2[i] =y_pos2[i]/cnt2[i]
		else:
			x_pos2[i]= last_x_pos2[i]
			y_pos2[i]=last_y_pos2[i]
			
		if frame==0:
			x_pos[6]=x_pos[1]
			y_pos[6]=y_pos[1]	
			x_pos2[6]=x_pos2[1]
			y_pos2[6]=y_pos2[1]		

		if frame>0:
			disparity = distance(x_pos[i],y_pos[i],x_pos2[i],y_pos2[i]) *pixel_size
			if disparity>0:			
				z_pos[i] = f_length * baseline/disparity	
				dist[i] = math.sqrt(z_pos[i]*z_pos[i]-(baseline/2)*(baseline/2))		
			else:
				z_pos[i] = last_z_pos[i]

			if distance3d(x_pos[i],y_pos[i],z_pos[i],last_x_pos[i],last_y_pos[i],last_z_pos[i])!=0:
				di_cos_x[i]= math.acos(((x_pos[i]-last_x_pos[i])/pixel_size)/distance3d(x_pos[i],y_pos[i],z_pos[i],last_x_pos[i],last_y_pos[i],last_z_pos[i]))/3.14*180
				di_cos_y[i]= math.acos(((y_pos[i]-last_y_pos[i])/pixel_size)/distance3d(x_pos[i],y_pos[i],z_pos[i],last_x_pos[i],last_y_pos[i],last_z_pos[i]))/3.14*180
				di_cos_z[i]= math.acos((z_pos[i]-last_z_pos[i])/distance3d(x_pos[i],y_pos[i],z_pos[i],last_x_pos[i],last_y_pos[i],last_z_pos[i]))/3.14*180
			if abs(di_cos_x[i]-last_di_cos_x[i])>5 or abs(di_cos_y[i]-last_di_cos_y[i])>5 or abs(di_cos_z[i]-last_di_cos_z[i])>5 :
				ufo1_id=1
			else:
				ufo1_id=0

			if distance3d(x_pos2[i],y_pos2[i],z_pos[i],last_x_pos2[i],last_y_pos2[i],last_z_pos[i])!=0:
				di_cos_x2[i]= math.acos(((x_pos2[i]-last_x_pos2[i])/pixel_size)/distance3d(x_pos2[i],y_pos2[i],z_pos[i],last_x_pos2[i],last_y_pos2[i],last_z_pos[i]))/3.14*180
				di_cos_y2[i]= math.acos(((y_pos2[i]-last_y_pos2[i])/pixel_size)/distance3d(x_pos2[i],y_pos2[i],z_pos[i],last_x_pos2[i],last_y_pos2[i],last_z_pos[i]))/3.14*180
				di_cos_z2[i]= math.acos((z_pos[i]-last_z_pos[i])/distance3d(x_pos2[i],y_pos2[i],z_pos[i],last_x_pos2[i],last_y_pos2[i],last_z_pos[i]))/3.14*180	
			
			if abs(di_cos_x2[i]-last_di_cos_x2[i])>5 or  abs(di_cos_y2[i]-last_di_cos_y2[i])>5 or abs(di_cos_z2[i]-last_di_cos_z2[i])>5 :
				ufo2_id=1
			else:
				ufo2_id =0
			
			ufo[i] = ufo[i] +ufo1_id*ufo2_id

			dis_data=str(frame)+"	"+colors[i]+"		"+str("{:.2e}".format(dist[i])) 						
			print (dis_data)
			f.writelines(dis_data+"\n")		

		last_x_pos[i]=x_pos[i]
		last_y_pos[i]=y_pos[i]
		last_z_pos[i]=z_pos[i]
		last_di_cos_x[i]= di_cos_x[i] 
		last_di_cos_y[i]= di_cos_y[i] 
		last_di_cos_z[i]= di_cos_z[i] 
		last_di_cos_x2[i]= di_cos_x2[i] 
		last_di_cos_y2[i]= di_cos_y2[i] 
		last_di_cos_z2[i]= di_cos_z2[i] 		

		cv2.rectangle(image,(int(x_pos[i])-5,int(y_pos[i])-5), (int(x_pos[i])+5,int(y_pos[i])+5), (0,0,255), 1)
		cv2.rectangle(image2,(int(x_pos2[i])-5,int(y_pos2[i])-5), (int(x_pos2[i])+5,int(y_pos2[i])+5), (0,0,255), 1)
		
	print("_________________________________________")
	f.writelines("_________________________________________"+"\n")

	cv2.imshow('image-left', image) #show the image 	
	cv2.imshow('image-right', image2) #show the image 
	cv2.waitKey(1)

uf_str="UFO:"	
max_val=0
max_id=0
for j in range(len(ufo)):
	if ufo[j]>max_val:
		max_val=ufo[j]
		max_id=j

uf_str ="UFO: "+colors[max_id]	
print(uf_str)
f.writelines(uf_str+"\n")
f.close()
#cv2.destroyAllWindows()
