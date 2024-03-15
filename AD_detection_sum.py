'''
Code for classify the image for 4 different class of Dimential Patient
Here we map the image into a 2 by 2 Inertia Tensor, then calculate Eigen Value from it 
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

############# Creating Position Matrix)############################

Im_mat = cv2.imread("../../Alzheimer_s_Dataset/train/ModerateDemented/moderateDem0.jpg", 0)
print(type(Im_mat))
print(np.sum(Im_mat))
#######
mod_dem_filename = "modereateDemo_sum.txt"
non_dem_filename = "nonDemo_sum.txt"
mild_dem_filename = "mildDemo_sum.txt"
very_mild_dem_filename = "verymildDemo_sum.txt"

############################Moderate Dimentia################################
if not os.path.exists(mod_dem_filename):  # if not file exist then run
    f_mod = open(mod_dem_filename, "w")
    for i in range(50):
        Im_mat = cv2.imread(
            f"././Alzheimer_s_Dataset/train/ModerateDemented/moderateDem{i}.jpg", 0
        )

        f_mod.write(f"{np.sum(Im_mat)}\n")
    ##
    f_mod.close()

############################Non Dimentia################################
if not os.path.exists(non_dem_filename):  # if not file exist then run
    f_non = open(non_dem_filename, "w")
    for i in range(500):
        Im_mat = cv2.imread(
            f"././Alzheimer_s_Dataset/train/NonDemented/nonDem{i}.jpg", 0
        )

        f_non.write(f"{np.sum(Im_mat)}\n")
    f_non.close()

############################Mild Dimentia################################
if not os.path.exists(mild_dem_filename):
    f_mild = open(mild_dem_filename, "w")
    for i in range(500):
        Im_mat = cv2.imread(
            f"././Alzheimer_s_Dataset/train/MildDemented/mildDem{i}.jpg", 0
        )

        f_mild.write(f"{np.sum(Im_mat)}\n")
    ##
    f_mild.close()

############################Very Mild Dimentia################################
if not os.path.exists(very_mild_dem_filename):
    f_verymild = open(very_mild_dem_filename, "w")
    for i in range(500):
        Im_mat = cv2.imread(
            f"././Alzheimer_s_Dataset/train/VeryMildDemented/verymildDem{i}.jpg", 0
        )


        f_verymild.write(f"{np.sum(Im_mat)}\n")
    ##
    f_verymild.close()


###############################Scatter Plot ####################################
mod_dem_eigdata = np.loadtxt(mod_dem_filename)
non_dem_eigdata = np.loadtxt(non_dem_filename)
mild_dem_eigdata = np.loadtxt(mild_dem_filename)
very_mild_dem_eigdata = np.loadtxt(very_mild_dem_filename)

print('Here is the mean and std for mod dem')
print(np.average(mod_dem_eigdata),np.std(mod_dem_eigdata))
print('Here is the mean and std for non dem')
print(np.average(non_dem_eigdata),np.std(non_dem_eigdata))
print('Here is the mean and std for mild dem')
print(np.average(mild_dem_eigdata),np.std(mild_dem_eigdata))
print('Here is the mean and std for very mild dem')
print(np.average(very_mild_dem_eigdata),np.std(very_mild_dem_eigdata))

#plt.axhline(np.average(non_dem_eigdata) ,label = 'non_avg',color='red')
##plt.plot(non_dem_eigdata,'o',label = 'non')
#plt.axhline(np.average(very_mild_dem_eigdata),label = 'very mild_avg',color='green')
##plt.plot(very_mild_dem_eigdata,'o',label = 'very mild')
#plt.axhline(np.average(mild_dem_eigdata),label = 'mild_avg',color='blue')
##plt.plot(mild_dem_eigdata,'o',label = 'mild')
#plt.axhline(np.average(mod_dem_eigdata),label = 'moderate_avg',color='black')
##plt.plot(mod_dem_eigdata,'o',label = 'moderate')
#plt.legend()
#plt.savefig("scatter_sum.jpg")
#plt.show()
########
########plt.scatter(
########    mod_dem_eigdata[:, 0],
########    mod_dem_eigdata[:, 1],
########    c="blue",
########    label="moderate",
########)
########plt.scatter(
########    non_dem_eigdata[:, 0],
########    non_dem_eigdata[:, 1],
########    c="red",
########    label="non",
########)
########plt.scatter(
########    mild_dem_eigdata[:, 0],
########    mild_dem_eigdata[:, 1],
########    c="green",
########    label="mild",
########)
########plt.scatter(
########    very_mild_dem_eigdata[:, 0],
########    very_mild_dem_eigdata[:, 1],
########    c="black",
########    label="very_mild",
########)
########plt.legend()
########plt.savefig("scatter.jpg")
########plt.show()
