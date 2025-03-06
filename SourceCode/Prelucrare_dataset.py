import math

import numpy as np
import pandas as pd
import os
import cv2 as openCV
import pathlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pydicom

import multiprocessing
import Canny
from Canny import canny_edge_detection
from scipy.spatial import Delaunay
import trimesh

# fiti siguri ca puneti locatia care trebuie
FOLDER_DIRECTORY = "C:/Reconstruire/DataSet_Sliced/ST000001/SE000004/" #locatia fisierului unde tinem minte fisierele .jpg si .dmc
# vedeti sa aveti un   /   la final !!!!!!!!!!!!!!!!!!!!!!!!!!!!!



class data_set:
    Data_set_file = [None] * 100 #initializare nula a setului de date
    Data_numbers = 0 # tinem minte cate seturi de date nenule sunt in fisier


    # set de date cu punctele de interes pentru export (.obj)
    Point_Array  = []

    # set de date cu triunghiurile pentru export (.obj)
    Triangle_Array =[]

    ###################################################################################

    #documentatie imread cu formate compatibile https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html
    def show_image(self, image_directory):  #modul de afisat imagini cu openCV
        if os.path.isfile(image_directory):
            img = None
            img = openCV.imread(image_directory) #citim datele imaginii
            openCV.imshow('image', img)  # deschidem o fereastra cu imaginea
            openCV.waitKey(0) # asteptam o actiune
            openCV.destroyAllWindows() #distrugem fereasta
        else:
            print("Image directory does not exist.")

    #trebuie o functie ptr afisat greyscale si .dcm ptr debugging etc
    def plotter(self, image):
        plt.imshow(image, cmap=plt.get_cmap('gray')) #incarcam imaginea in plotter
        plt.show() # afisam imaginea

    # functie ptr convertire grayscale .jpg/.png
    def rgb2grayscale(self, rgb):
        gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]) #convertire rgb to grayscale, mai jos am pus linkul cu formula
        #https://saturncloud.io/blog/how-to-convert-an-image-to-grayscale-using-numpy-arrays-a-comprehensive-guide/
        #print(gray.shape) # alta chestie pentru debugging
        #self.plotter(gray) # pentru debugging
        return gray #returneaza grayscale-ul imaginilor

    #functie pentru extragere date .jpg, .png sau .dcm etc. din fisierul corespunzator
    def extract_data_files_file_type(self, folder_directory, file_type):
        if os.path.exists(folder_directory):  # verificam daca exista fisier cu acea locatie
            i=0
            for name in os.listdir(folder_directory):
                if pathlib.Path(folder_directory+name).suffix == file_type: #verific daca sufixul este file_type
                    #self.show_image(folder_directory+name) #pentru debugging
                    if file_type == ".dcm":
                        dicom = pydicom.dcmread(folder_directory + name)  # citim special tipul .dcm
                        self.Data_set_file[i] = dicom.pixel_array
                        self.Data_set_file[i] = canny_edge_detection(self.Data_set_file[i])
                        #self.plotter(self.Data_set_file[i]) #afisare imagine dataset, ptr debugging
                    else:
                        image = openCV.imread(folder_directory + name) #citim special tipurile .png si .jpg
                        edge = np.uint8(self.rgb2grayscale(image))
                        #self.Data_set_file[i] = np.uint8(self.rgb2grayscale(image)) #tinem minte ca un np array greyscale-ul imaginii
                        #self.plotter(edge) #afisare imagine dataset, ptr debugging
                        self.Data_set_file[i] = openCV.Canny(edge, 100, 200)
                        #self.plotter(self.Data_set_file[i])
                    i=i+1
            self.Data_numbers = i
        else:
            print("File directory does not exist.")

    # functie pentru generare fisier .obj
    def output_organ_data_3D(self, save_folder):
        if os.path.exists(save_folder):
            object_file = open(save_folder + "test.obj", "w") # salvez undeva un fisier .obj
            for Points in self.Point_Array:  # scriu punctele din array in fisier
                object_file.write("v " + str(Points[0]) + " " + str(Points[1]) + " " + str(Points[2]) + '\n')

            for Triangles in self.Triangle_Array: # scriu triunghiurile din array-ul de triunghiuri in acelasi fisier
                object_file.write("f " + str(Triangles[0]) + " " + str(Triangles[1]) + " " + str(Triangles[2]) + '\n')
            object_file.close() # inchid fisierul pentru a nu pierde date
        else:
            print("Save directory does not exist.")

    # returneaza tipul setului de date (pentru debugging)
    def getfile_extension(self):
        return self.__file_extension

    def point_condition(self, point_vector, point, distance): # conditie pentru a folosi un punct
        if len(point_vector)==0: #daca vectorul e nul, evident bagam o valoare
            return True
        ok=True
        #+math.pow(test[2]-point[2],2)
        for test in point_vector: #testam toate punctele precedente
            if math.sqrt(math.pow(test[0]-point[0],2)+ math.pow(test[1]-point[1],2)+math.pow(test[2]-point[2],2)) <=distance:
               ok=False # pur si simplu o distanta
        return ok

    def ReLU(self, n):
        if n>0:
            return n
        return 0




    def convolve(self, img, step, dim):
        curstep = 0
        weights = [[random.rand() for e in range(dim)] for e in range(dim)]
        newImg = [None] * 100



    def max_pooling(self):
        potato;

    def average_pooling(self):
        potatopower;

    def plotter_3D(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        
        #  pune practic punctele in figura
        ax.scatter(x, y, z)

        # scrie un X mare langa axa X
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # open window
        plt.show()


    def layer_reconstruction(self,number,scale_x, scale_y, distance, z_cur):
        layer_points = [] # punem cate un strat pe rand
        #self.plotter(self.Data_set_file[number])
        for x in range(self.Data_set_file[number].shape[0]):
            for y in range(self.Data_set_file[number].shape[1]): #iteram pixelii imaginilor
                coord = [scale_x*(512-x),scale_y*(512-y),z_cur] # coordonate
                if self.Data_set_file[number][x][y] == 255 and self.point_condition(layer_points, coord,distance): #verificam daca e pixel puternic
                    layer_points.append(coord) # scriem in vectorul de puncte
        print(layer_points)
        self.Point_Array.append(layer_points) # introducem punctele stratului


    #introduce punctele care sunt edges in vectorul de puncte
    def edge_to_3D(self, scale_x, scale_y, dim_z, distance): #avem anumite scalari si dimensiunea in z
        #layer_points = [] #punem cate un strat pe rand

        #Processes = []

        step_z = dim_z/self.Data_numbers # calculam pasul necesar pentru fiecare iteratie
        z_cur = -dim_z/2 #setam primul punct de plecare

      #  for k in range(self.Data_numbers):
           # if __name__ == '__main__':
           #     p = multiprocessing.Process(target = self.layer_reconstruction, args = [k, scale_x, scale_y, distance, z_cur])
           #     p.start()
           #     Processes.append(p)
           # z_cur = z_cur + step_z  # trecem la imaginea urmatoare

        #for process in Processes:
           # process.join()

        for k in range(self.Data_numbers): #interam setul de date
            for x in range(self.Data_set_file[k].shape[0]):
                for y in range(self.Data_set_file[k].shape[1]): #iteram pixelii imaginilor
                    coord = [scale_x*(512-x),scale_y*(512-y),z_cur] # coordonate
                    if self.Data_set_file[k][x][y] == 255 and self.point_condition(self.Point_Array, coord,distance): #verificam daca e pixel puternic
                        self.Point_Array.append(coord) # scriem in vectorul de puncte
            z_cur = z_cur + step_z #trecem la imaginea urmatoare
        #self.Point_Array.append(layer_points) # introducem punctele stratului
    #daca e nevoie, scriu un ghid cum am ajuns la aceste formule (sunt din capul meu after all)

    def point_to_mesh(self):
        triangles = Delaunay(self.Point_Array[:, :2]).simplices
        mesh = trimesh.Trimesh(vertices=self.Point_Array, faces=triangles)
        mesh.export("output.stl")


    # constructor pentru fisierul nostru
    def __init__(self, folder_directory, file_type):
        self.__file_extension = file_type # setam un format in care avem setul de date, si il declaram privat
        # pentru a nu-l putea schimba (din greseala)
        self.extract_data_files_file_type(folder_directory,file_type)
        self.edge_to_3D(1,1,400,50)
        print(self.Point_Array)  # pentru debugging
        text_file_path = 'data.txt'
        np.savetxt(text_file_path, self.Point_Array, delimiter=' ')

        self.point_to_mesh()
        print(self.Point_Array) # pentru debugging


MYDATA_BASE = data_set(FOLDER_DIRECTORY, ".jpg")
MYDATA_BASE.output_organ_data_3D("C:/Reconstruire/")