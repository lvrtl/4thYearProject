# -*- coding: utf-8 -*-
#Modelling intreactions for pair of atoms
import csv
import numpy as np
from numpy.linalg import matrix_power
import matplotlib.pyplot as plt
from numpy import linalg as LA
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDirectionArrows
import os
from scipy.optimize import leastsq

class point:
    def __init__(self,x,y,z,ID):
        self.x = x
        self.y = y
        self.z = z
        self.v = np.array([x,y,z])
        self.ID = ID
        self.neighbours = []
        self.surfaceAtom = False

class lattice:
     
    @staticmethod
    def combinations(elements):
        """
        return list of sum of permutations of -1, 0 and 1 times each element
        in input
        """
        
        output = []
        for i in range(3**len(elements)):
            if i == 0:
                nums = [-1*elements[0]]
            else:
                nums = []
                while i:
                    i, r = divmod(i, 3)
                    nums.append((r-1)*elements[len(nums)])
            while len(nums) < len(elements):
                nums.append(-1*elements[len(nums)])
            output.append(np.sum(np.array(nums),0))
        return(output)

    @staticmethod
    def read(name):
        """
        reads a csv file and outputs a list of lists, holding data as strings
        """
        with open(name, newline='') as f:
            reader = csv.reader(f)
            return list(reader)
        
    @staticmethod
    def write(data, name):
        """
        Writes a csv file.
        data is stored as list of lists
        """
        if os.path.exists(name) == False:
            open(name,'x')
        with open(name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)
    

    @staticmethod
    def DeStringify(ListOfListOfStrings):
        """
        converts a list of list of strings to an array of floats
        """
        FloatArray= np.zeros([len(ListOfListOfStrings),len(ListOfListOfStrings[0])])
        for i in range(len(ListOfListOfStrings)):
            for j in range(len(ListOfListOfStrings[0])):
                FloatArray[i,j] = float(ListOfListOfStrings[i][j])
        return(FloatArray)        

    @staticmethod
    def Indexer(InputArray):
        """
        converts an array, formatted as a set of ID's corresponding to a value in
        each column, to an array, with the number of dimensions matching the
        number of ID's 
        """
        dim = len(InputArray[0]) - 1
        OutputArray = np.zeros(dim*[(1+int(np.max(InputArray[:,0:dim])))])

        if dim == 1:
            for element in InputArray:
                OutputArray[int(element[0])] = element[1]
        elif dim == 2:
            for element in InputArray:
                OutputArray[int(element[0]),int(element[1])] = element[2]
                OutputArray[int(element[1]),int(element[0])] = element[2]
        else:
            for element in InputArray:
                OutputArray[int(element[0]),int(element[1])
                            ,int(element[2])] = element[3]
                OutputArray[int(element[2]),int(element[1])
                            ,int(element[0])] = element[3]
        return(OutputArray)
        
    def ChainData(length,ID):
        """
        Produce data file for a chain of atoms
        """
        data = [[x,0,0,ID] for x in range(length)]
        with open('nChainData.csv', 'w', newline = '') as f:
            # create the csv writer
            writer = csv.writer(f)
            # write a row to the csv file
            writer.writerows(data)
            
    def ZincblendeCubic(a,ID1,ID2,filename):
        """
        Produce unit cell data for a zinc blende cubic strucure.
        a is lattice constant along any edge
        ID1 is ID of atom in corner
        ID2 is ID of atom displaced from corner
        """
        data = []
        data.append([0,0,0,ID1])
        data.append([0.5*a,0.5*a,0,ID1])
        data.append([0,0.5*a,0.5*a,ID1])
        data.append([0.5*a,0,0.5*a,ID1])
        data.append([0.25*a,0.25*a,0.25*a,ID2])
        data.append([0.75*a,0.75*a,0.25*a,ID2])
        data.append([0.25*a,0.75*a,0.75*a,ID2])
        data.append([0.75*a,0.25*a,0.75*a,ID2])
        lattice.write(data,filename)
        
            
    def LatticeData(UnitCell,filename,a,n):
        """
        Produce lattice data file and boundary conditions for any unit cell.
        
        UnitCell is a lattice object.
        a1, a2 and a3 are lattice constant vectors (numpy vectors)
        n1, n2 and n3 are the periodicity distances of the 3 lattice constants.
        returns boundary condition vectors.
        writes lattice data file to filename.
        """
        data = []
        #loop over every unit cell reproduction.
        for i in range(n[0]):
            for j in range(n[1]):
                for k in range(n[2]):
                    for point in UnitCell.points:
                        data.append((point.v+i*a[0]+j*a[1]+k *a[2]).tolist() + [point.ID])
        
        #write to file
        lattice.write(data,filename)
        
        #return boundary condition vectors
        return [a[0]*n[0],a[1]*n[1],a[2]*n[2]]
            
    def __init__(self,PointDataFileName,MassDataFileName,StretchDataFileName
                 ,LengthDataFileName,BendDataFileName,AngleDataFileName, BoundaryConditions = None):
        """
        
        PointData file is formatted as first row x, second row y, third row z, 
        last row ID: assign each to a lattice point object. ID is atomic number-1
        
        MassIndex: row of masses
        
        BondStretchData: Indexes bond stretching factor of BS(IJ). 
        Format is ID I, ID J, length.
        
        IdealLengthData: Stores ideal bond length of BS(IJ). 
        Format is ID I, ID J, length.
        
        IdealAngleData: Stores ideal angle of BB(JIK). 
        Format is ID J, ID I, ID K, angle.
        
        BondBendingData: Stores bond bending factor of BB(JIK). 
        Format is ID J, ID I, ID K, angle.
        
        Point Data is converted into point objects
        
        All other data types are converted to arrays of floats, and then 
        an array of dimension equal to the number of ID's it has stores
        the data
        """

        data = self.read(PointDataFileName)

        self.points = [point(float(data[i][0]),float(data[i][1]),float(data[i][2]),
                             int(data[i][3])) for i in range(len(data))]
        self.DynamicalMatrix = np.zeros((3*len(self.points),3*len(self.points)))  
        
#        self.pointCoordinates = []
#        for point in self.points:
#            self.pointCoordinates.append
        
        data = self.read(MassDataFileName)
        self.MassIndex= self.Indexer(self.DeStringify(data))
        
        data = self.read(LengthDataFileName)
        self.IdealLengthIndex= self.Indexer(self.DeStringify(data))
        
        data = self.read(StretchDataFileName)
        self.BondStretchingIndex= self.Indexer(self.DeStringify(data))

        data = self.read(BendDataFileName)
        self.BondBendingIndex= self.Indexer(self.DeStringify(data))
        
        data = self.read(AngleDataFileName)
        self.IdealAngleIndex= self.Indexer(self.DeStringify(data))
        
        self.BoundaryConditions = BoundaryConditions
        
        if (self.BoundaryConditions is None) == False:
            #produce matrix of boundary condition vectors
            self.BoundaryConditionsCombinations = self.combinations(BoundaryConditions)

            
        
    def neighbourscalc(self, bondlength):
        for PointToCheck in self.points: 
            for neighbour in set(self.points) - {PointToCheck}: #check against every other lattice point
                if self.DistanceObj(PointToCheck,neighbour,None)<=bondlength:
                    PointToCheck.neighbours.append(neighbour)
    
    def DistV(self, Point1, Point2, Axis = None):
        """
        calculates the shortest vector between 2 points, as vectors
        """
        
        if self.BoundaryConditions is None:
            if Axis == None:
                return(Point2 - Point1) 
            else:
                return(Point2[Axis]-Point1[Axis])
        else:
            return(sorted([Point2-Point1-i
                   for i in self.BoundaryConditionsCombinations],
                   key = np.linalg.norm)[0])

        
    def Distance(self, Point1, Point2, Axis = None):
        """
        gets the Distance between 2 lattice points
        Point1 and Point2 are 2 lattice points / their indexes in the lattice
        object's list of points
        AreObjects is a boolean for whether the points are objects or their
        indexes(default)
        Axis defines whether the Distance is taken along a specific axis
        """
        
        return(np.linalg.norm(self.DistV(self.points[Point1].v , self.points[Point2].v,)[Axis]))
            
    def DistanceObj(self, Point1, Point2, Axis = None):
        """
        gets the Distance between 2 lattice points
        Point1 and Point2 are 2 lattice points / their indexes in the lattice
        object's list of points
        AreObjects is a boolean for whether the points are objects or their
        indexes(default)
        Axis defines whether the Distance is taken along a specific axis
        """
        
        return(np.linalg.norm(self.DistV(Point1.v,Point2.v)[Axis]))

   
    def innerObj(self,j,i,k):
        """
        Returns r(ij).r(ik)
        """
        return(np.inner(self.DistV(i.v,j.v)),self.DistV(i.v,k.v))
    
    def inner(self,j,i,k, AreObjects = False):
        
        """
        Returns r(ij).r(ik)
        """

        return np.inner(self.DistV(
                self.points[i].v,self.points[j].v),
                self.DistV(self.points[i].v,self.points[k].v))
    
    def matcalc(self):
        
        MassMatrix = lambda arg1: np.sqrt(arg1)*np.identity(3)
        
        #anon fn to generate mass matrix for given input mass
        
        for i in range(len(self.points)):
            pointI = self.points[i]
            for m in range(3):
                    for n in range(3):
                        if m == n:       
                            
                            """
                            partial^2 U
                            /(partial r^i_m)^2
                            """
                            for pointJ in pointI.neighbours:
                                #BS term
                                j = self.points.index(pointJ)
                                self.DynamicalMatrix[3*i+m,3*i+n]+=3/2 *(
                                    self.BondStretchingIndex[pointI.ID,pointJ.ID] 
                                    * (self.Distance(i,j)**2
                                    - self.IdealLengthIndex[pointI.ID,pointJ.ID]**2 
                                    + 2 * self.Distance(i,j,m)**2
                                    ) / (self.IdealLengthIndex[pointI.ID,pointJ.ID])**2)
                                
                                #BB term 1                     
                                for pointK in set(pointI.neighbours) - {pointJ}:
                                    
                                    k = self.points.index(pointK)
                                    self.DynamicalMatrix[3*i+m,3*i+n]+=3/8*(
                                        self.BondBendingIndex[pointJ.ID,pointI.ID,pointK.ID]*
                                        (-2*(np.cos(np.pi/180*self.IdealAngleIndex[pointJ.ID,pointI.ID,pointK.ID]))
                                        +(2*self.inner(j,i,k)
                                        +(self.Distance(j,i,m)+self.Distance(k,i,m))**2)
                                        /self.IdealLengthIndex[pointI.ID,pointJ.ID]
                                        /self.IdealLengthIndex[pointI.ID,pointK.ID]))
                                
                                #BB term 2
                                for pointK in set(pointJ.neighbours) - {pointI}:
                                    k = self.points.index(pointK)
                                    self.DynamicalMatrix[3*i+m,3*i+n]+=3/4*(
                                        self.BondBendingIndex[pointI.ID,pointJ.ID,pointK.ID]*
                                        (-1*(np.cos(np.pi/180*(self.IdealAngleIndex[pointI.ID,pointJ.ID,pointK.ID])))
                                        +(self.inner(i,j,k)
                                        +self.Distance(j,k,m)**2)
                                        /self.IdealLengthIndex[pointJ.ID,pointI.ID]
                                        /self.IdealLengthIndex[pointJ.ID,pointK.ID]))
                                
 
                        else:
                            """
                            partial^2 U
                            /(partial r^i_m)(\partial r^i_n)
                            m=/=n
                            """
                            for neighbour in self.points[i].neighbours:
                                j = self.points.index(neighbour)
                                
                                self.DynamicalMatrix[3*i+m,3*i+n]+=3*(
                                    self.BondStretchingIndex[pointI.ID,pointJ.ID] 
                                    * self.Distance(i,j,m)
                                    * self.Distance(i,j,n)
                                    / self.IdealLengthIndex[pointI.ID,pointJ.ID]**2)
                                
                                #BB term 1                     
                                for pointK in set(self.points[i].neighbours) - {pointJ}:
                                    k = self.points.index(pointK)
                                    self.DynamicalMatrix[3*i+m,3*i+n]+=3/8*(
                                        self.BondBendingIndex[pointJ.ID,pointI.ID,pointK.ID]
                                        *(self.Distance(j,i,n)+self.Distance(k,i,n))
                                        *(self.Distance(j,i,m)+self.Distance(k,i,m))
                                        /self.IdealLengthIndex[pointI.ID,pointJ.ID]
                                        /self.IdealLengthIndex[pointI.ID,pointK.ID])
                                
                                #BB term 2
                                for pointK in set(self.points[j].neighbours) - {pointI}:
                                    k = self.points.index(pointK)
                                    self.DynamicalMatrix[3*i+m,3*i+n]+=3/4*(
                                        self.BondBendingIndex[pointI.ID,pointJ.ID,pointK.ID]
                                        *self.Distance(j,k,m)
                                        *self.Distance(j,k,n)
                                        /self.IdealLengthIndex[pointJ.ID,pointI.ID]
                                        /self.IdealLengthIndex[pointJ.ID,pointK.ID])
            
            #incororate mass
            self.DynamicalMatrix[3*i:(3*i+3),3*i:(3*i+3)] = (
                        np.dot(np.dot(matrix_power(
                        MassMatrix(self.MassIndex[self.points[i].ID]),-1),
                        self.DynamicalMatrix[3*i:(3*i+3),3*i:(3*i+3)]),
                        matrix_power(MassMatrix(self.MassIndex[self.points[i].ID]),-1))) 
                
            for pointJ in pointI.neighbours:
                """
                nearest neighbour interaction terms
                """
                j = self.points.index(pointJ)
                            
                #if already calculated symmetrically, reassign.
                if np.any(self.DynamicalMatrix[3*i:(3*i+3),3*j:(3*j+3)]):
                    self.DynamicalMatrix[3*i:3*i+3,3*j:3*j+3] = np.transpose(
                            self.DynamicalMatrix[3*j:(3*j+3),3*i:(3*i+3)] )
                else:
                    #looping for each dimension permutation:
                    for m in range(3):
                        for n in range(3):
                            if m == n:
                                #calculation for same dimension; will add on each term 
                                #for readability and ease of looping
                                """
                                partial^2 U
                                /(partial r^i_m)(\partial r^j_m)
                                j in nn(i)
                                """
                                
                                #Bond Stretching
                                self.DynamicalMatrix[3*i+m,3*j+n] += -3/2 * (
                                    self.BondStretchingIndex[pointI.ID,pointJ.ID] 
                                    * (self.Distance(i,j)**2 - 
                                    (self.IdealLengthIndex[pointI.ID,pointJ.ID])**2 
                                    + 2 * self.Distance(i,j,m)**2) 
                                    / (self.IdealLengthIndex[pointI.ID,pointJ.ID])**2)
                                
                                #Bond Bending #1
                                for pointK in set(pointI.neighbours) - {pointJ}:
                                    k = self.points.index(pointK)
                                    self.DynamicalMatrix[3*i+m,3*j+n] += 3/4* (
                                        self.BondBendingIndex[pointJ.ID,pointI.ID,pointK.ID] 
                                        *(np.cos(np.pi/180*self.IdealAngleIndex[pointJ.ID,pointI.ID,pointK.ID])
                                        +(-self.inner(j,i,k)-self.Distance(i,k,m)
                                          *(self.Distance(i,k,m)+self.Distance(i,j,m)))
                                        /self.IdealLengthIndex[pointI.ID,pointJ.ID]
                                        /self.IdealLengthIndex[pointI.ID,pointK.ID]))
                                        
                                #Bond Bending #2
                                for pointK in set(pointJ.neighbours) - {pointI}:
                                    k = self.points.index(pointK)
                                    self.DynamicalMatrix[3*i+m,3*j+n] += 3/4* (
                                        self.BondBendingIndex[pointI.ID,pointJ.ID,pointK.ID] 
                                        *(np.cos(np.pi/180*self.IdealAngleIndex[pointI.ID,pointJ.ID,pointK.ID])
                                        +(-self.inner(i,j,k)-self.Distance(j,k,m)
                                          *(self.Distance(j,k,m)+self.Distance(j,i,m)))
                                        /self.IdealLengthIndex[pointJ.ID,pointI.ID]
                                        /self.IdealLengthIndex[pointJ.ID,pointK.ID]))
                                    
                                #Bond bending #3
                                for pointK in set(pointI.neighbours).intersection(set(pointJ.neighbours)):
                                    k = self.points.index(pointK)
                                    self.DynamicalMatrix[3*i+m,3*j+n] += 3/4*(
                                        self.BondBendingIndex[pointI.ID,pointK.ID,pointJ.ID] 
                                        *(-np.cos(np.pi/180*self.IdealAngleIndex[pointI.ID,pointK.ID,pointJ.ID])
                                        +(self.inner(i,j,k)+self.Distance(k,i,m)
                                          *self.Distance(k,j,m))
                                        /self.IdealLengthIndex[pointK.ID,pointI.ID]
                                        /self.IdealLengthIndex[pointK.ID,pointJ.ID]))
                            else:
                                #calculation for different dimension
                                """
                                partial^2 U
                                /(partial r^i_m)(\partial r^j_n)
                                j in nn(i)
                                m=/=n
                                """
                                self.DynamicalMatrix[3*i+m,3*j+n] += -3 * (
                                    self.BondStretchingIndex[pointI.ID, pointJ.ID] 
                                    * self.Distance(i,j,m) 
                                    * self.Distance(i,j,n)
                                    / (self.IdealLengthIndex[pointI.ID,pointJ.ID])**2)
                                
                                #Bond Bending #1
                                for pointK in set(pointI.neighbours) - {pointJ}:
                                    k = self.points.index(pointK)
                                    self.DynamicalMatrix[3*i+m,3*j+n] += -3/4* (
                                        self.BondBendingIndex[pointJ.ID,pointI.ID,pointK.ID] 
                                        *self.Distance(i,k,n)
                                        *(self.Distance(i,k,m)+self.Distance(i,j,m))
                                        /self.IdealLengthIndex[pointI.ID,pointJ.ID]
                                        /self.IdealLengthIndex[pointI.ID,pointK.ID])
                                    
                                #Bond Bending #2
                                for pointK in set(pointJ.neighbours) - {pointI}:
                                    k = self.points.index(pointK)
                                    self.DynamicalMatrix[3*i+m,3*j+n] += -3/4* (
                                        self.BondBendingIndex[pointI.ID,pointJ.ID,pointK.ID] 
                                        *self.Distance(j,k,m)
                                        *(self.Distance(j,k,n)+self.Distance(j,i,n))
                                        /self.IdealLengthIndex[pointJ.ID,pointI.ID]
                                        /self.IdealLengthIndex[pointJ.ID,pointK.ID])
                                    
                                #Bond bending #3
                                for pointK in set(pointI.neighbours).intersection(set(pointJ.neighbours)):
                                    k = self.points.index(pointK)
                                    self.DynamicalMatrix[3*i+m,3*j+n] += 3/4*(
                                        self.BondBendingIndex[pointI.ID,pointK.ID,pointJ.ID] 
                                        *self.Distance(k,i,n)
                                        *self.Distance(k,j,m)
                                        /self.IdealLengthIndex[pointK.ID,pointI.ID]
                                        /self.IdealLengthIndex[pointK.ID,pointJ.ID])
                            
                    #include mass in dynamical matrix
                    self.DynamicalMatrix[3*i:(3*i+3),3*j:(3*j+3)
                        ] = np.dot(np.dot(matrix_power(MassMatrix(self.MassIndex[self.points[i].ID]),-1),
                        self.DynamicalMatrix[3*i:(3*i+3),
                        3*j:(3*j+3)]),
                        matrix_power(MassMatrix(self.MassIndex[self.points[j].ID]),-1))
            

            """
            partial^2 U
            /(partial r^i_m)(\partial r^j_m)
            m not in nn(i) U i
            """    
            
            #work out set of points that are not nearest neighbours or i,
            #but neighbours of nearest neighbours of i 
            pointJset = set([]) 
            for pointK in pointI.neighbours:
                for pointJ in set(pointK.neighbours)- {pointI} - set(pointI.neighbours):
                    pointJset.add(pointJ)
            for pointJ in pointJset:
                j = self.points.index(pointJ)
                #if already calculated symmetrically, reassign.
                if np.any(self.DynamicalMatrix[3*i:(3*i+3),3*j:(3*j+3)]):
                    self.DynamicalMatrix[3*i:3*i+3,3*j:3*j+3] = np.transpose(
                            self.DynamicalMatrix[3*j:(3*j+3),3*i:(3*i+3)] )
                else:
                    for pointK in set(pointJ.neighbours).intersection(set(pointI.neighbours)):
                        k= self.points.index(pointK)
                        for m in range(3):
                                for n in range(3):
                                    if m == n:
                                        self.DynamicalMatrix[3*i+m,3*j+n] += 3/4*(
                                            self.BondBendingIndex[pointI.ID,pointK.ID,pointJ.ID]
                                            *(-(np.cos(np.pi/180*self.IdealAngleIndex[pointI.ID,pointK.ID,pointJ.ID]))
                                            +(self.inner(i,k,j)
                                            + self.Distance(k,j,m)*self.Distance(k,i,m)
                                            )/self.IdealLengthIndex[pointK.ID,pointI.ID]
                                            /self.IdealLengthIndex[pointK.ID,pointJ.ID]))
                                            
                                    else:
                                        self.DynamicalMatrix[3*i+m,3*j+n] += 3/4*(
                                            self.BondBendingIndex[pointI.ID,pointK.ID,pointJ.ID]
                                            *self.Distance(k,j,m)*self.Distance(k,i,n)
                                            /self.IdealLengthIndex[pointK.ID,pointI.ID]
                                            /self.IdealLengthIndex[pointK.ID,pointJ.ID])
                                        
                    #include mass in dynamical matrix
                    self.DynamicalMatrix[3*i:(3*i+3),3*j:(3*j+3)
                        ] = np.dot(np.dot(matrix_power(MassMatrix(self.MassIndex[self.points[i].ID]),-1),
                        self.DynamicalMatrix[3*i:(3*i+3),
                        3*j:(3*j+3)]),
                        matrix_power(MassMatrix(self.MassIndex[self.points[j].ID]),-1))
                
    def EigSolve(self):
        eigenvals, vectors = LA.eig(self.DynamicalMatrix)
#        self.eigenvals = np.around(np.absolute(eigenvals),3)
        self.eigenvals = eigenvals
        self.w = np.sqrt(np.round(np.real(self.eigenvals)))
#        self.vectors = np.around(np.absolute((vectors)),3)
        self.vectors = vectors
                       
    def EigPlot2D(self,ax,ValToPlot,param_dict,ArrowWidth = 0.1):
        #
        """
        graph for eigenvectors which are ALL orthogonal to one axis
        
        Parameters
        ----------
        ax : Axes
            The axes to draw to
            
        ValtoPlot: eigenvalue/eigenvectors to plot
    
        param_dict : dict
           Dictionary of kwargs to pass to ax.plot
        
        Returns
        -------
        out : list
            list of artists added
        """
        CoordDict = {0: 'x', 1: 'y', 2: 'z'}
        OrthAxis = None

        #decide on orthogonal axis based on which has zero vals

        for i in range(3):            
            if np.any([self.vectors[i + 3*x,ValToPlot] for x in range(len(self.points))])== False:
                OrthAxis = i
         
        if OrthAxis == None:
            raise TypeError("Eigenvectors are not orthogonal to x, y or z.")
            return
        
        Axes = list({0,1,2} - {OrthAxis})
        
        out = []
        #plot arrows then points on top
        for i in range(len(self.points)):
            pointI = self.points[i]
            #only plot arrow if there is non-zero eigenvector in the plane
            if np.any([self.vectors[[3*i + Axes[0],3*i + Axes[1]],ValToPlot]])== True:
                out.append(plt.arrow(pointI.v[Axes[0]],
                          pointI.v[Axes[1]],
                          self.vectors[3*i + Axes[0],ValToPlot]/4,
                          self.vectors[3*i + Axes[1],ValToPlot]/4,
                          width = ArrowWidth))
            
            out.append(plt.plot(pointI.v[Axes[0]], pointI.v[Axes[1]],
                     **param_dict))
            
        simple_arrow = AnchoredDirectionArrows(ax.transAxes,
                                               CoordDict[Axes[0]], 
                                               CoordDict[Axes[1]],
                                               color = 'black')
        
        out.append(ax.add_artist(simple_arrow))
        
        textXLoc = (max([self.points[a].v[0] for a in range(len(self.points))
        ])) * 0.8
        out.append(plt.text(textXLoc,0.4,'w^2 =' + str(np.round(self.eigenvals[ValToPlot],2))))
        
        return out
            
    def EigPlotPhonon(self,ax,ValToPlot,param_dict,ArrowWidth = 0.1):
        #
        """
        graph for visualising phonons in single dimension
        
        Parameters
        ----------
        ax : Axes
            The axes to draw to
            
        ValtoPlot: eigenvalue/eigenvectors to plot
    
        param_dict : dict
           Dictionary of kwargs to pass to ax.plot
        
        Returns
        -------
        out : list
            list of artists added
        """
        #CoordDict = {0: 'x', 1: 'y', 2: 'z'}
        Axes = [0,1,2]

        #decide on orthogonal axis based on which has zero vals

        for i in range(3):            
            if np.any([self.vectors[i + 3*x,ValToPlot] for x in range(len(self.points))])== False:
                Axes.remove(i)
         
        if Axes[0] == None:
            raise TypeError("Eigenvectors are multidimensional.")
            return
        
        out = []
        #plot arrows then points on top

        #only plot arrow if there is non-zero eigenvector in the plane
        plt.plot([point.v[Axes[0]] for point in self.points],
                      [self.vectors[3*i + Axes[0],ValToPlot] for i in range(len(self.points))])
            
        
        textXLoc = (max([self.points[a].v[0] for a in range(len(self.points))
        ])) * 0.8
        out.append(plt.text(textXLoc,0.4,'w^2 =' + str(np.round(self.eigenvals[ValToPlot],2))))
        
        return out
    
    
    def PhononBasis(self):
        '''
        creates PhononBasis attribute based on a vectors and n values
        if an.am = 0
        also creates index of which k-vectors are on path, and numbers them
        '''
        if self.BoundaryConditions is None:
            R = []
            for i in range(len(self.a)):
                R.append(self.a[i]*self.n[i])
        else:
            R = self.BoundaryConditions
        #assume orthogonal
        if len(self.a) == 3:
            self.reciprocals = 2*np.pi*np.linalg.inv(np.concatenate([a[None,:] for a in self.a]))
         
        #generate set of k vectors        
        k = np.array([2 * np.pi * R[0] / np.linalg.norm(R[0])**2 * n for n in range(self.n[0])])
                                            

        
        k = np.concatenate([k,np.array([
                2 * np.pi * -1 * R[0] / np.linalg.norm(R[0])**2 * n
                + 2 * np.pi * 3 * R[1] / np.linalg.norm(R[1])**2 * n
                + self.reciprocals[0] for n in range(int(self.n[0]/4))])])
        
#        print(np.array([
#                2 * np.pi * -1 * R[0] / np.linalg.norm(R[0])**2 * n
#                + 2 * np.pi * 3 * R[1] / np.linalg.norm(R[1])**2 * n
#                + self.reciprocals[0] for n in range(int(self.n[0]/4))]))
        
        k = np.concatenate([k,np.array([
                3/4 * self.reciprocals[0] + 3/4 * self.reciprocals[1]
                - 2 * np.pi * 3 * R[0] / np.linalg.norm(R[0])**2 * n
                - 2 * np.pi * 3 * R[1] / np.linalg.norm(R[1])**2 * n
                for n in range(int(self.n[0]/4))])])
                                            
#        print(np.array([
#                3/4 * self.reciprocals[0] + 3/4 * self.reciprocals[1]
#                - 2 * np.pi * 3 * R[0] / np.linalg.norm(R[0])**2 * n
#                - 2 * np.pi * 3 * R[1] / np.linalg.norm(R[1])**2 * n
#                for n in range(int(self.n[0]/4))]))
                                            
        k = np.concatenate([k,np.array([
                2 * np.pi * R[0] / np.linalg.norm(R[0])**2 * n
                + 2 * np.pi * R[1] / np.linalg.norm(R[1])**2 * n
                for n in range(1, 1+int(self.n[0]/2))])])
                                            
#        print(np.array([
#                2 * np.pi * R[0] / np.linalg.norm(R[0])**2 * n
#                + 2 * np.pi * R[1] / np.linalg.norm(R[1])**2 * n
#                for n in range(1, 1+int(self.n[0]/2))])) 
#                                    
#        print(k)
        
        self.kPath = np.linspace(0,1,len(k)).tolist()
#        self.kPath = np.concatenate([np.array(element*24) for element in kPathTrue])
        self.k = k
        
        #generate U
        u = np.zeros([24,len(self.points),3])
        axes = np.array([[1,0,0],[0,1,0],[0,0,1]])
        for axis in range(3):
            for j in range(8):
                for i in range(len(self.points)):
                    u[8*axis+j,i] = axes[axis] * ( (i+j )% 8 == 0)
                    
        self.u = u
        self.write(u,'u.csv')
        #store phonon basis as
        self.PhononBasisV = np.zeros([len(self.k) * len(u),len(self.vectors)],dtype = np.complex64)
        
        
        for kIndex in range(len(self.k)):
            for uIndex in range(len(u)):
                for PointIndex in range(len(self.points)):
                    self.PhononBasisV[len(u) * kIndex + uIndex,
                        3*PointIndex:3*PointIndex + 3] = u[uIndex,PointIndex
                        ] * np.exp(1j * np.dot(self.k[kIndex], 
                        self.points[PointIndex].v[0]))
                #normalise
        
        for i in range(len(self.PhononBasisV)):
            if LA.norm(self.PhononBasisV[i]) != 0 :
                    self.PhononBasisV[i] = self.PhononBasisV[i] / LA.norm(self.PhononBasisV[i])
        self.write(self.PhononBasisV,'pb.csv')
        
        
        
    def EigenProjection(self):
        '''
        project eigenvectors onto basis
        '''
        self.projections = np.zeros([len(self.vectors),len(self.PhononBasisV)])
        for index in range(len(self.vectors)):
            self.projections[index] = np.round(np.power
                             (np.abs(np.inner(self.PhononBasisV,self.vectors[:,index])),2),3)
        
    def kDecomposition(self, ax, index):
        '''
        create figure of intensity of eigenstate against k-value.

        Parameters
        ----------
        ax : Axes
            The axes to draw to
            
        index: index of eigenvalue to plot
    
        param_dict : dict
           Dictionary of kwargs to pass to ax.plot
        
        Returns
        -------
        out : list
            list of artists added
        '''
        out = [plt.scatter(self.Kx,self.projections[index])]
        out.append(plt.title('phonon '+ str(index)+', w = ' +str(np.round(self.w[index],2))))
        out.append(plt.xlabel('k'))
        out.append(plt.ylabel('I'))
        return out
        
    def kPlot(self,fig,**kwargs ):
        '''
        plot along k-path
        return list if figures
        '''
        figs = []
        self.scatterx = []
        self.scattery = []
        for i in range(len(self.projections)):
            for j in range(len(self.projections[0])):
                if self.projections[i,j] > 0.05:
                    self.scatterx.append(self.kPath[int(j/24)])
                    self.scattery.append(self.w[i])
        
        print(len(self.scatterx))
        print(len(self.scattery))
        
        figs.append(plt.scatter(self.scatterx,self.scattery))
        figs.append(plt.title(str(self.n) + 'phonon dispersion' ))
        figs.append(plt.xlabel('k'))
        figs.append(plt.ylabel('w'))
        return 
    
    def uPlot(self,k,fig):
        '''
        plot u against w for a given k
        return figure
        '''
        figs = []
        scatterx = []
        scattery = []
        for u in range(24):
            for i in range(len(self.projections)):
                    scatterx.append(u)
                    scattery.append(self.projections[i,u+24*k])
                    
        figs.append(plt.scatter(scatterx,scattery))
        figs.append(plt.title(str(self.n) + 'phonon dispersion: k =' +str(self.k[k]) ))
        figs.append(plt.xlabel('u'))
        figs.append(plt.ylabel('decomposition'))
        
    
if __name__ == "__main__":
    
    a = 565
    aVectors = np.array([[a,0,0],[0,a,0],[0,0,a]])
    n = np.array([4,4,4])
    #generate gallium arsenide unit cell
    lattice.ZincblendeCubic(a,30,32,'UnitCell.csv')
    UnitCell = lattice('UnitCell.csv','MassIndex.csv',
                            'BondStretchingData.csv','BondLengthData.csv',
                            'BondBendingData.csv','IdealAngleData.csv',
                            aVectors)
    
    BC = lattice.LatticeData(UnitCell,'GalliumArsenide.csv',aVectors,n)
    GaAs = lattice('GalliumArsenide.csv','MassIndex.csv',
                            'BondStretchingData.csv','BondLengthData.csv',
                            'BondBendingData.csv','IdealAngleData.csv',BC)
    
    GaAs.n = n
    GaAs.a = aVectors
    GaAs.neighbourscalc(np.sqrt(3)/4 * a)
    GaAs.matcalc()
    GaAs.EigSolve()
    GaAs.PhononBasis()
    GaAs.EigenProjection()
    
    #define directory for storing plots
    #first, get directory
    real_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(real_path)
    if GaAs.BoundaryConditions is None:
        chaintype = 'chainFin'
    else:
        chaintype = 'chainPeriodic'
    
    #create directory
    ImgDir = dir_path +  '\\' + 'GaAS' + str(n) + chaintype + '\\'
    if os.path.exists(ImgDir) == False:
        os.mkdir(ImgDir)
    
    
    #store dynamical matrix to file
    mat_path = ImgDir+'\\'+"DynamicalMatrix"+ '.csv'
    
    if os.path.exists(mat_path) == False:
            open(mat_path,'x')
                  
    with open(mat_path, 'w',newline = '') as f:
        # create the csv writer
        writer = csv.writer(f)
    
        # write a row to the csv file
        writer.writerows(np.ndarray.tolist(GaAs.DynamicalMatrix))


    """     
    plotting data
    """
    
    fig, ax = plt.subplots(1, 1)
    GaAs.kPlot(fig )
    plt.title(str(GaAs.n) + 'GaAs, periodic')
    graph_path_phonon = ImgDir+'\\'+'phonons.svg'
    if os.path.exists(graph_path_phonon) == False:
        open(graph_path_phonon,'x')
    plt.savefig(graph_path_phonon)    
    plt.close()
    
    
    fig, ax = plt.subplots(1, 1)
    for i in range(len(GaAs.k)):
        GaAs.uPlot(i ,fig)
        plt.title(str(GaAs.n) + 'GaAs, periodic')
        graph_path_phonon = ImgDir+'\\'+'uPlot' + str(i) +  '.svg'
        if os.path.exists(graph_path_phonon) == False:
            open(graph_path_phonon,'x')
        plt.savefig(graph_path_phonon)    
        plt.close()
    
#    fig, ax = plt.subplots(1, 1)
#    for i in range(len(chain.projections)):
#        for k in range(len(chain.projections[0])):
#            if chain.projections[i,k] !=0:
#                plt.scatter(chain.Kx[k],abs(chain.w[i]),color = 'b')
#    plt.xlabel('k')
#    plt.ylabel('w')
#    plt.title(str(chainlength)+' atom chain, periodic boundary conditions')
#    graph_path_phonon = ImgDir+'\\'+'phonons.svg'
#    if os.path.exists(graph_path_phonon) == False:
#        open(graph_path_phonon,'x')
#    plt.savefig(graph_path_phonon)    
#    plt.close()
#    
#    #eigenstate decomposition
#    fig, ax = plt.subplots(1,1)
#    for i in range(10):
#        fig, ax = plt.subplots(1, 1)
#        chain.kDecomposition(ax,i)
#        graph_path_eigenstateDecomp = ImgDir+'\\'+str(i)+ 'eigenstateDecomp.svg'
#        if os.path.exists(graph_path_eigenstateDecomp) == False:
#            open(graph_path_eigenstateDecomp,'x')
#        plt.savefig(graph_path_eigenstateDecomp)
#        plt.close()
#
#    plt.close()
#    #motion plot
#    for i in range(3*chainlength):
#        fig, ax = plt.subplots(1, 1)
#        chain.EigPlot2D(ax,i,{'marker': 'o','color' : 'r'},0.02)
#        plt.ylim([-0.5,0.5])
#        plt.xlim([-0.5,max([chain.points[a].v[0] for a in range(len(chain.points))])+0.5])
#        graph_path_motion = ImgDir+'\\'+str(i)+ 'motion.s vg'
#        if os.path.exists(graph_path_motion) == False:
#            open(graph_path_motion,'x')
#        plt.savefig(graph_path_motion)
#        plt.close()
#       
#    #eigenstate plot
#    for i in range(3*chainlength):
#        fig, ax = plt.subplots(1, 1)
#        chain.EigPlotPhonon(ax,i,{'marker': 'o','color' : 'r'},0.02)
#        plt.ylim([-1,1])
#        plt.xlim([-0.5,max([chain.points[a].v[0] for a in range(len(chain.points))])+0.5])
#        graph_path_phonon = ImgDir+'\\'+str(i)+ 'phonon.svg'
#        if os.path.exists(graph_path_phonon) == False:
#            open(graph_path_phonon,'x')
#        plt.savefig(graph_path_phonon)
#        plt.close()