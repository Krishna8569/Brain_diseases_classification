'''
Code for classify the image for 4 different class of Dimential Patient
Here we map the image into a 2 by 2 Inertia Tensor, then calculate Eigen Value from it 
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

#############
############# Creating Position Matrix)############################

Im_mat = cv2.imread("./Alzheimer_s_Dataset/train/ModerateDemented/moderateDem0.jpg", 0)
#print(Im_mat)
plt.imshow(Im_mat)
plt.colorbar()
plt.show()
nx = Im_mat.shape[0]
ny = Im_mat.shape[1]
x = np.linspace(-1, 1, ny)
y = np.linspace(1 * nx / ny, -1 * nx / ny, nx)
pos_mat = np.zeros([nx, ny, 2])
for i in range(nx):
    for j in range(ny):
        pos_mat[i, j, 0] = x[j]
        pos_mat[i, j, 1] = y[i]

#################### Defineing file name#################################

mod_dem_filename = "modereateDemo_eig.txt"
non_dem_filename = "nonDemo_eig.txt"
mild_dem_filename = "mildDemo_eig.txt"
very_mild_dem_filename = "verymildDemo_eig.txt"

############################Moderate Dimentia################################
if not os.path.exists(mod_dem_filename):  # if not file exist then run
    f_mod = open(mod_dem_filename, "w")
    for i in range(50):
        Im_mat = cv2.imread(
            f"./Alzheimer_s_Dataset/train/ModerateDemented/moderateDem{i}.jpg", 0
        )

        I = np.zeros([2, 2])
        for i in range(nx):
            for j in range(ny):
                m = Im_mat[i, j]
                rx = pos_mat[i, j, 0]
                ry = pos_mat[i, j, 1]
                I[0, 0] += m * ry**2
                I[0, 1] += -m * rx * ry
                I[1, 1] += m * rx**2

        I[1, 0] = I[0, 1]

        ## finding eigen value

        eig_val_I, eig_vec_I = np.linalg.eig(I)
        f_mod.write(f"{eig_val_I[0]}  {eig_val_I[1]}\n")
        print("eigen value of inertia tensor of the given figure is =", eig_val_I, end = '\r')
    ##
    f_mod.close()

############################Non Dimentia################################
if not os.path.exists(non_dem_filename):  # if not file exist then run
    f_non = open(non_dem_filename, "w")
    for i in range(500):
        Im_mat = cv2.imread(
            f"./Alzheimer_s_Dataset/train/NonDemented/nonDem{i}.jpg", 0
        )

        I = np.zeros([2, 2])
        for i in range(nx):
            for j in range(ny):
                m = Im_mat[i, j]
                rx = pos_mat[i, j, 0]
                ry = pos_mat[i, j, 1]
                I[0, 0] += m * ry**2
                I[0, 1] += -m * rx * ry
                I[1, 1] += m * rx**2

        I[1, 0] = I[0, 1]

        ## finding eigen value

        eig_val_I, eig_vec_I = np.linalg.eig(I)
        f_non.write(f"{eig_val_I[0]}  {eig_val_I[1]}\n")
        print("eigen value of inertia tensor of the given figure is =", eig_val_I,end='\r')
    f_non.close()

############################Mild Dimentia################################
if not os.path.exists(mild_dem_filename):
    f_mild = open(mild_dem_filename, "w")
    for i in range(500):
        Im_mat = cv2.imread(
            f"./Alzheimer_s_Dataset/train/MildDemented/mildDem{i}.jpg", 0
        )

        I = np.zeros([2, 2])
        for i in range(nx):
            for j in range(ny):
                m = Im_mat[i, j]
                rx = pos_mat[i, j, 0]
                ry = pos_mat[i, j, 1]
                I[0, 0] += m * ry**2
                I[0, 1] += -m * rx * ry
                I[1, 1] += m * rx**2

        I[1, 0] = I[0, 1]

        ## finding eigen value

        eig_val_I, eig_vec_I = np.linalg.eig(I)
        f_mild.write(f"{eig_val_I[0]}  {eig_val_I[1]}\n")
        # print("eigen value of inertia tensor of the given figure is =", eig_val_I)
        print("eigen value of inertia tensor of the given figure is =", eig_val_I,end='\r')
    ##
    f_mild.close()

############################Very Mild Dimentia################################
if not os.path.exists(very_mild_dem_filename):
    f_verymild = open(very_mild_dem_filename, "w")
    for i in range(500):
        Im_mat = cv2.imread(
            f"./Alzheimer_s_Dataset/train/VeryMildDemented/verymildDem{i}.jpg", 0
        )

        I = np.zeros([2, 2])
        for i in range(nx):
            for j in range(ny):
                m = Im_mat[i, j]
                rx = pos_mat[i, j, 0]
                ry = pos_mat[i, j, 1]
                I[0, 0] += m * ry**2
                I[0, 1] += -m * rx * ry
                I[1, 1] += m * rx**2

        I[1, 0] = I[0, 1]

        ## finding eigen value

        eig_val_I, eig_vec_I = np.linalg.eig(I)
        f_verymild.write(f"{eig_val_I[0]}  {eig_val_I[1]}\n")
        # print("eigen value of inertia tensor of the given figure is =", eig_val_I)
        print("eigen value of inertia tensor of the given figure is =", eig_val_I,end='\r')
    ##
    f_verymild.close()


###plot##################################Scatter Plot ####################################
###plot###mod_dem_eigdata = np.loadtxt(mod_dem_filename)
###plot###non_dem_eigdata = np.loadtxt(non_dem_filename)
###plot###mild_dem_eigdata = np.loadtxt(mild_dem_filename)
###plot###very_mild_dem_eigdata = np.loadtxt(very_mild_dem_filename)
###plot###
###plot####plt.plot(np.max(mod_dem_eigdata,axis=1) - np.min(mod_dem_eigdata,axis=1),label = 'moderate')
###plot###plt.axhline(np.average(np.max(mod_dem_eigdata,axis=1) - np.min(mod_dem_eigdata,axis=1)),label = 'moderate_avg',color='black')
###plot####plt.plot(np.max(non_dem_eigdata,axis=1) - np.min(non_dem_eigdata,axis=1),label = 'non')
###plot###plt.axhline(np.average(np.max(non_dem_eigdata,axis=1) - np.min(non_dem_eigdata,axis=1)),label = 'non_avg',color='red')
###plot####plt.plot(np.max(mild_dem_eigdata,axis=1) - np.min(mild_dem_eigdata,axis=1),label = 'mild')
###plot###plt.axhline(np.average(np.max(mild_dem_eigdata,axis=1) - np.min(mild_dem_eigdata,axis=1)),label = 'mild_avg',color='blue')
###plot####plt.plot(np.max(very_mild_dem_eigdata,axis=1) - np.min(very_mild_dem_eigdata,axis=1),label = 'very mild')
###plot###plt.axhline(np.average(np.max(very_mild_dem_eigdata,axis=1) - np.min(very_mild_dem_eigdata,axis=1)),label = 'very mild_avg',color='green')
###plot###plt.legend()
###plot###plt.show()
###plot#####print(np.max(mod_dem_eigdata,axis=1)) #- np.min(mod_dem_eigdata))
###plot####plt.scatter(
###plot####    mod_dem_eigdata[:, 0],
###plot####    mod_dem_eigdata[:, 1],
###plot####    c="blue",
###plot####    label="moderate",
###plot####)
###plot####plt.scatter(
###plot####    non_dem_eigdata[:, 0],
###plot####    non_dem_eigdata[:, 1],
###plot####    c="red",
###plot####    label="non",
###plot####    alpha = 0.1,
###plot####)
###plot####plt.scatter(
###plot####    mild_dem_eigdata[:, 0],
###plot####    mild_dem_eigdata[:, 1],
###plot####    c="green",
###plot####    label="mild",
###plot####    alpha = 0.2,
###plot####)
###plot####plt.scatter(
#    very_mild_dem_eigdata[:, 0],
#    very_mild_dem_eigdata[:, 1],
#    c="black",
#    label="very_mild",
#)
#plt.legend()
#plt.savefig("scatter.jpg")
#plt.show()
