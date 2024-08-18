	/*
		CS 6023 Assignment 3. 
		Do not make any changes to the boiler plate code or the other files in the folder.
		Use cudaFree to deallocate any memory not in usage.
		Optimize as much as possible.
	*/

	#include "SceneNode.h"
	#include <queue>
	#include "Renderer.h"
	#include <stdio.h>
	#include <string.h>
	#include <cuda.h>
	#include <chrono>

	__global__
	void scenemake2(int offset, int xdir, int ydir, int amount, int *dGlobalCoordinatesX, int *dGlobalCoordinatesY, int *dCsr, int *dOffset, int numOfChild){
		int id=blockIdx.x*blockDim.x+threadIdx.x;
		if(id<numOfChild){
			id+=offset;
			atomicAdd(&dGlobalCoordinatesX[dCsr[id]],xdir*amount);
			atomicAdd(&dGlobalCoordinatesY[dCsr[id]],ydir*amount);
			int numOfThreads=dOffset[dCsr[id]+1]-dOffset[dCsr[id]];
			int numberOfBlocks = (numOfThreads+blockDim.x-1)/blockDim.x;
			scenemake2<<<numberOfBlocks, numOfThreads>>>(dOffset[dCsr[id]], xdir, ydir, amount,dGlobalCoordinatesX, dGlobalCoordinatesY, dCsr, dOffset,numOfThreads);
		}
	}
	__global__ void scenemake(int meshNum , int xdir, int ydir, int amount,int* dGlobalCoordinatesX,int* dGlobalCoordinatesY,int* dCsr,int* dOffset, int numOfChild){
		int numOfThreads=dOffset[meshNum+1]-dOffset[meshNum];
		int numberOfBlocks=(numOfThreads+blockDim.x-1)/blockDim.x;
		if(ydir==0)	atomicAdd(&dGlobalCoordinatesX[meshNum],xdir*amount);
		else 	atomicAdd(&dGlobalCoordinatesY[meshNum],ydir*amount);
		scenemake2<<<numberOfBlocks,numOfThreads>>>(dOffset[meshNum],xdir,ydir,amount,dGlobalCoordinatesX,dGlobalCoordinatesY,dCsr,dOffset,numOfThreads);    
	}
	__global__
	void renderx(int *mesh_i,int dGlobalX_i, int dGlobalY_i, int dOpacity_i, int *dframeOp,int frameSizeX, int frameSizeY, int *dFinalPng){
		int mesh_ind=blockIdx.x*blockDim.x+threadIdx.x;
		int inFrame_x=dGlobalX_i+blockIdx.x;
		int inFrame_y=dGlobalY_i+threadIdx.x;
		int inFrame_ind=inFrame_x*frameSizeY+inFrame_y;
		if(inFrame_x>=0 && inFrame_y>=0 && inFrame_y<frameSizeY && inFrame_x<frameSizeX && dframeOp[inFrame_ind]<dOpacity_i){
			dFinalPng[inFrame_ind]=mesh_i[mesh_ind];
			dframeOp[inFrame_ind]=dOpacity_i;
		}
	}


	void readFile (const char *fileName, std::vector<SceneNode*> &scenes, std::vector<std::vector<int> > &edges, std::vector<std::vector<int> > &translations, int &frameSizeX, int &frameSizeY) {
		/* Function for parsing input file*/

		FILE *inputFile = NULL;
		// Read the file for input. 
		if ((inputFile = fopen (fileName, "r")) == NULL) {
			printf ("Failed at opening the file %s\n", fileName) ;
			return ;
		}

		// Input the header information.
		int numMeshes ;
		fscanf (inputFile, "%d", &numMeshes) ;
		fscanf (inputFile, "%d %d", &frameSizeX, &frameSizeY) ;
		

		// Input all meshes and store them inside a vector.
		int meshX, meshY ;
		int globalPositionX, globalPositionY; // top left corner of the matrix.
		int opacity ;
		int* currMesh ;
		for (int i=0; i<numMeshes; i++) {
			fscanf (inputFile, "%d %d", &meshX, &meshY) ;
			fscanf (inputFile, "%d %d", &globalPositionX, &globalPositionY) ;
			fscanf (inputFile, "%d", &opacity) ;
			currMesh = (int*) malloc (sizeof (int) * meshX * meshY) ;
			for (int j=0; j<meshX; j++) {
				for (int k=0; k<meshY; k++) {
					fscanf (inputFile, "%d", &currMesh[j*meshY+k]) ;
				}
			}
			//Create a Scene out of the mesh.
			SceneNode* scene = new SceneNode (i, currMesh, meshX, meshY, globalPositionX, globalPositionY, opacity) ; 
			scenes.push_back (scene) ;
		}

		// Input all relations and store them in edges.
		int relations;
		fscanf (inputFile, "%d", &relations) ;
		int u, v ; 
		for (int i=0; i<relations; i++) {
			fscanf (inputFile, "%d %d", &u, &v) ;
			edges.push_back ({u,v}) ;
		}

		// Input all translations.
		int numTranslations ;
		fscanf (inputFile, "%d", &numTranslations) ;
		std::vector<int> command (3, 0) ;
		for (int i=0; i<numTranslations; i++) {
			fscanf (inputFile, "%d %d %d", &command[0], &command[1], &command[2]) ;
			translations.push_back (command) ;
		}
	}


	void writeFile (const char* outputFileName, int *hFinalPng, int frameSizeX, int frameSizeY) {
		/* Function for writing the final png into a file.*/
		FILE *outputFile = NULL; 
		if ((outputFile = fopen (outputFileName, "w")) == NULL) {
			printf ("Failed while opening output file\n") ;
		}
		
		for (int i=0; i<frameSizeX; i++) {
			for (int j=0; j<frameSizeY; j++) {
				fprintf (outputFile, "%d ", hFinalPng[i*frameSizeY+j]) ;
			}
			fprintf (outputFile, "\n") ;
		}
	}


	int main (int argc, char **argv) {
		
		// Read the scenes into memory from File.
		const char *inputFileName = argv[1] ;
		int* hFinalPng ; 

		int frameSizeX, frameSizeY ;
		std::vector<SceneNode*> scenes ;
		std::vector<std::vector<int> > edges ;
		std::vector<std::vector<int> > translations ;
		readFile (inputFileName, scenes, edges, translations, frameSizeX, frameSizeY) ;
		hFinalPng = (int*) malloc (sizeof (int) * frameSizeX * frameSizeY) ;
		
		// Make the scene graph from the matrices.
		Renderer* scene = new Renderer(scenes, edges) ;

		// Basic information.
		int V = scenes.size () ;
		int E = edges.size () ;
		int numTranslations = translations.size () ;

		// Convert the scene graph into a csr.
		scene->make_csr () ; // Returns the Compressed Sparse Row representation for the graph.
		int *hOffset = scene->get_h_offset () ;  
		int *hCsr = scene->get_h_csr ();
		int *hOpacity = scene->get_opacity () ; // hOpacity[vertexNumber] contains opacity of vertex vertexNumber.
		int **hMesh = scene->get_mesh_csr () ; // hMesh[vertexNumber] contains the mesh attached to vertex vertexNumber.
		int *hGlobalCoordinatesX = scene->getGlobalCoordinatesX () ; // hGlobalCoordinatesX[vertexNumber] contains the X coordinate of the vertex vertexNumber.
		int *hGlobalCoordinatesY = scene->getGlobalCoordinatesY () ; // hGlobalCoordinatesY[vertexNumber] contains the Y coordinate of the vertex vertexNumber.
		int *hFrameSizeX = scene->getFrameSizeX () ; // hFrameSizeX[vertexNumber] contains the vertical size of the mesh attached to vertex vertexNumber.
		int *hFrameSizeY = scene->getFrameSizeY () ; // hFrameSizeY[vertexNumber] contains the horizontal size of the mesh attached to vertex vertexNumber.

		auto start = std::chrono::high_resolution_clock::now () ;
		
		// Code begins here.
		// Do not change anything above this comment.
		
		memset(hFinalPng, 0, sizeof (int) * frameSizeX * frameSizeY);

		int *dOffset;
		int *dCsr;
		// int *dOpacity;
		// int **dMesh;
		int *dGlobalCoordinatesX;
		int *dGlobalCoordinatesY;
		// int *dFrameSizeX;
		// int *dFrameSizeY;
		int *mesh_i;
		int *dFinalPng;

		cudaMalloc(&dOffset, sizeof (int) * (V+1)) ;
		cudaMalloc(&dGlobalCoordinatesX, sizeof (int) * V) ;
		cudaMalloc(&dGlobalCoordinatesY, sizeof (int) * V) ;
		cudaMalloc(&dCsr, sizeof (int) * E) ;
		cudaMalloc(&dFinalPng, sizeof (int) * frameSizeX * frameSizeY);
		
		cudaMemcpy(dOffset, hOffset, sizeof (int) * (V+1), cudaMemcpyHostToDevice);
		cudaMemcpy(dGlobalCoordinatesX, hGlobalCoordinatesX, sizeof (int) * V, cudaMemcpyHostToDevice);
		cudaMemcpy(dGlobalCoordinatesY, hGlobalCoordinatesY, sizeof (int) * V, cudaMemcpyHostToDevice);
		cudaMemcpy(dCsr, hCsr, sizeof (int) * E, cudaMemcpyHostToDevice);
		
		int xdir, ydir;
		for(int i=0;i<numTranslations;i++){
			int meshNum=translations[i][0];
			int command=translations[i][1];
			int amount=translations[i][2];
			if(command==0){//up
				xdir=-1;
				ydir=0;
			}
			else if(command==1){//down
				xdir=1;
				ydir=0;
			}
			else if(command==2){//left
				xdir=0;
				ydir=-1;
			}
			else{//right
				xdir=0;
				ydir=1;
			}
			scenemake<<<1,1>>>(meshNum, xdir, ydir, amount,dGlobalCoordinatesX, dGlobalCoordinatesY, dCsr, dOffset, 1);
		}	
		cudaDeviceSynchronize();5
		cudaFree(dOffset);
		cudaFree(dCsr);

		cudaMemcpy(hGlobalCoordinatesX, dGlobalCoordinatesX, sizeof (int) * V, cudaMemcpyDeviceToHost);
		cudaMemcpy(hGlobalCoordinatesY, dGlobalCoordinatesY, sizeof (int) * V, cudaMemcpyDeviceToHost);
		
		cudaFree(dGlobalCoordinatesX);
		cudaFree(dGlobalCoordinatesY);
		
		int *hframeOp = (int*) malloc (sizeof(int) * frameSizeX * frameSizeY) ;
		memset(hframeOp, INT_MIN, sizeof(int) * frameSizeX * frameSizeY);
		
		int *dframeOp;
		cudaMalloc(&dframeOp , sizeof(int) * frameSizeX * frameSizeY) ;
		cudaMemcpy(dframeOp,hframeOp,sizeof(int) * frameSizeX * frameSizeY,cudaMemcpyHostToDevice);
		
		cudaMemcpy(dFinalPng, hFinalPng, sizeof (int) * frameSizeX * frameSizeY,cudaMemcpyHostToDevice);
		
		cudaMalloc(&mesh_i,10000*sizeof(int));
		for(int i=0;i<V;i++){
			cudaMemcpy(mesh_i, hMesh[i], sizeof (int) * hFrameSizeX[i]*hFrameSizeY[i], cudaMemcpyHostToDevice);
			renderx<<<hFrameSizeX[i],hFrameSizeY[i]>>>(mesh_i, hGlobalCoordinatesX[i], hGlobalCoordinatesY[i], hOpacity[i], dframeOp, frameSizeX, frameSizeY, dFinalPng);
		}
		cudaDeviceSynchronize();

		cudaMemcpy(hFinalPng, dFinalPng, sizeof (int) * frameSizeX * frameSizeY, cudaMemcpyDeviceToHost);

		// for(int i=0;i<frameSizeX;i++){
		// 	for(int j=0;j<frameSizeY;j++){
		// 		printf("%d\t",hFinalPng[i*frameSizeY+j]);
		// 	}
		// 	printf("\n");
		// }
		

		// Do not change anything below this comment.
		// Code ends here.

		auto end  = std::chrono::high_resolution_clock::now () ;

		std::chrono::duration<double, std::micro> timeTaken = end-start;

		printf ("execution time : %f\n", timeTaken) ;
		// Write output matrix to file.
		const char *outputFileName = argv[2] ;
		writeFile (outputFileName, hFinalPng, frameSizeX, frameSizeY) ;	

	}
