/*
 ============================================================================
 Name        : 2D_tundish.cu
 Author      : Jan Bohacek
 Version     :
 Copyright   : 
 Description : laminar flow in two-dimensional tundish in continuous casting
 ============================================================================
 */


#include <iostream>
//#include <stdio.h>
//#include <algorithm>
//#include <numeric>
#include <fstream>
#include <sstream>
#include <cstring>
//#include <ctime>
#include <math.h>

using namespace std;

typedef double T;	// precision of calculation

typedef struct {
	int Nx; 		// x-coordinate
	int Ny;			// y-coordinate
	T dx; 			// dx = dy = dz
} Dimensions;		// dimensions of geometry

typedef struct {
	int steps;		// number of timesteps (-)
	int maxIterSIMPLE; // maximum number of SIMPLE iterations
	T CFL;			// Courant number
	T dt;			// timestep size
	T UY;			// inlet velocity
	T ac;			// volume of cell divided by timestep
	T blocks;		// for dot product
	T blockSize;	// -||-
	T maxResU;		// stopping criterion for velocity calculation
	T maxResP;		// 					      pressure
	T maxResSIMPLE; //						  SIMPLE 
	T urfU;			// under-relaxation factor U
	T urfP;			// 						   P
} Parameters;		// simulation settings

typedef struct {
	T nu; 			// kinematic viscosity (m2/s)
	T rho;			// density
	T cp;			// specific heat
	T k;			// thermal conductivity
	T alpha;    	// thermal diffusivity (m2/s)
	T beta; 		// thermal expansion coefficient
} MaterialProperties;

// declare CPU fields
Dimensions dims;
Parameters params;
MaterialProperties liquid;

// cache constant CUDA fields
__constant__ Dimensions d_dims;
__constant__ Parameters d_params;
__constant__ MaterialProperties d_liquid;

#include "cpuFunctions.h"
#include "cudaFunctions.h"



int main()
{
	
		
	cout << "--flow in 2D tundish---" << endl;
		
	// geometry
	dims.Nx = 256;
	dims.Ny = 64;
	dims.dx = 0.001;
	
	// paramaters
	params.steps = 100;
	params.CFL = 1.0;
	params.UY = -0.5;
	params.dt = params.CFL * dims.dx / fabs(params.UY);
	params.ac = dims.dx*dims.dx/params.dt;
	params.blocks = 16;  
	params.blockSize = 128;  
	params.maxResU      = 1e-5;
	params.maxResP      = 1e-5;
	params.maxResSIMPLE = 1e-4;
	params.urfU         = 0.7;
	params.urfP         = 0.3;
	//params.maxIterSIMPLE=100;
	
	// material properties
	liquid.nu  = 0.000001;   // water 1e-6 m2/s
	liquid.rho = 1000;
	
	cout << "For Courant number of " << params.CFL << " the timestep size is " << params.dt << endl;
	
	// CPU fields
	T *ux;		// ux-component of velocity
	T *uy;		// uy
	T *p;		// pressure
	T *m;		// mass balance
	T *hrh,*hsg;	// dot products
	T rhNew, rhOld, sg, ap, bt;
	T endIter, rhNewSIMPLE;
	int iter, iterSIMPLE;
	//T *Temp; 	// temperature
	
	// GPU fields
	T *dux;		// ux-component of velocity
	T *duy;		// uy
	T *dp;		// pressure
	T *dm;      // mass balance
	T *dpo;		// old value of p pressure
	T *duxo;	// 				ux
	T *duyo;	// 				uy
	T *duxtemp;	// for swapping fields
	T *duytemp;
	T *duxc,*duxs,*duxw, *dkuxc, *dkuxs, *dkuxw; 	// Aux
	T *drx,*dqx,*dzx,*dpx;							// Aux
	T *duyc,*duys,*duyw, *dkuyc, *dkuys, *dkuyw; 	// Auy
	T *dry,*dqy,*dzy,*dpy;							// Auy
	T *dpc,*dps,*dpw, *dkpc, *dkps, *dkpw; 			// Ap
	T *drp,*dqp,*dzp,*dpp;							// Ap
	T *drh,*dsg;	// dot products
	//T *dTemp; 	// temperature
	
	// GPU parameters
	int THREADS_PER_BLOCK = 256;
	int BLOCKS = ((dims.Nx+2)*(dims.Ny+2)+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;	// larger in order to have BLOCKS*THREADS_PER_BLOCK > Nx*Ny
	// taken from CUDA by example
			
	// initialize fields 
	cpuInit(ux, uy, p, m, hrh, hsg);
	cudaInit(dux, duy, dp, dm, duxo, duyo, dpo,
			 duxc, duxs, duxw, dkuxc, dkuxs, dkuxw, 	// Aux
			 drx, dqx, dzx, dpx,						// Aux
			 duyc, duys, duyw, dkuyc, dkuys, dkuyw, 	// Auy
			 dry, dqy, dzy, dpy,						// Auy
			 dpc, dps, dpw, dkpc, dkps, dkpw, 			// Ap
			 drp, dqp, dzp, dpp,						// Ap
			 drh, dsg);
	
	// patch anything to dux
	//patchDux<<<BLOCKS,THREADS_PER_BLOCK>>>(dux);
	
	// patch anything to duy
	//patchDuy<<<BLOCKS,THREADS_PER_BLOCK>>>(duy);
	
	/*// copy back to host and save
	cudaMemcpy(ux, dux, sizeof(T)*(dims.Nx+2)*(dims.Ny+2), cudaMemcpyDeviceToHost);
	cudaMemcpy(uy, duy, sizeof(T)*(dims.Nx+2)*(dims.Ny+2), cudaMemcpyDeviceToHost);
	cudaMemcpy(p,   dp, sizeof(T)*dims.Nx    *(dims.Ny+2), cudaMemcpyDeviceToHost);
	saveDataInTime(ux, uy, p, m, (T)0, "testTundish");*/
	
	// Aux (x-component of velocity)
	Aux<<<BLOCKS,THREADS_PER_BLOCK>>>(duxc, duxs, duxw);
	// AuxInlet not necessary, velocity inlet condition ux=0 is the same as no slip condition at wall
	AuxOutlet<<<1,10>>>(duxc, 200);
	makeTNS1<<<BLOCKS,THREADS_PER_BLOCK>>>(dkuxc,dkuxs,dkuxw,duxc,duxs,duxw,dims.Nx-1,dims.Ny);
	
	// Auy (y-component of velocity)
	Auy<<<BLOCKS,THREADS_PER_BLOCK>>>(duyc, duys, duyw);
	//AuyInlet, AuyOutlet not necessary
	AuyOutlet<<<1,10>>>(duyc, 200);
	makeTNS1<<<BLOCKS,THREADS_PER_BLOCK>>>(dkuyc,dkuys,dkuyw,duyc,duys,duyw,dims.Nx,dims.Ny-1);
	
	// Ap (pressure)
	Ap<<<BLOCKS,THREADS_PER_BLOCK>>>(dpc, dps, dpw);
	ApOutlet<<<1,10>>>(dpc, 200); // Dirichlet, p=0
	makeTNS1<<<BLOCKS,THREADS_PER_BLOCK>>>(dkpc,dkps,dkpw,dpc,dps,dpw,dims.Nx,dims.Ny);
	
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	
	
	for (int miter=0; miter<params.steps; miter++) {
	
		// boundary conditions
		bcVelWallNoslip<<<BLOCKS,THREADS_PER_BLOCK>>>(dux, duy);	// no slip at walls
		bcVelInlet<<<1,10>>>(dux, duy, params.UY, 50);				// bcVelInlet<<<1,inletWidth>>>(dux, duy, velocity, first index);
		bcVelOutlet<<<1,10>>>(dux, duy, 200);						// bcVelOutlet<<<1,outletwidth>>>(dux, duy, first index);
		
		//swap old and new arrays for next timestep
		duxtemp = duxo; duxo = dux; dux = duxtemp;
		duytemp = duyo; duyo = duy; duy = duytemp;
		
		// advect horizontal and vertical velocity components
		advectUx<<<BLOCKS,THREADS_PER_BLOCK>>>(dux, duxo, duyo);
		advectUy<<<BLOCKS,THREADS_PER_BLOCK>>>(duy, duxo, duyo);
		
		// boundary conditions
		bcVelWallNoslip<<<BLOCKS,THREADS_PER_BLOCK>>>(dux, duy);	// no slip at walls
		bcVelInlet<<<1,10>>>(dux, duy, params.UY, 50);				// bcVelInlet<<<1,inletWidth>>>(dux, duy, velocity, first index);
		bcVelOutlet<<<1,10>>>(dux, duy, 200);
		
		// ************ BEGIN SIMPLE **********
		iterSIMPLE    = 0;
		rhNewSIMPLE   = 1;
		
		
		while (rhNewSIMPLE > params.maxResSIMPLE) {  //(iterSIMPLE < params.maxIterSIMPLE) {
		
			iterSIMPLE++;
		
			// ********** BEGIN solve UX **********
			duToDr<<<BLOCKS,THREADS_PER_BLOCK>>>(drx, dux,dims.Nx-1,dims.Ny);					// drx := dux
			SpMV<<<BLOCKS,THREADS_PER_BLOCK>>>(dqx,duxc,duxs,duxw,drx,dims.Nx-1,dims.Ny);		// q := Aux ux 
			b<<<BLOCKS,THREADS_PER_BLOCK>>>(drx,dims.Nx-1,dims.Ny);								// drx := bx
			// bxInlet not necessary as ux=0 there
			bpx<<<BLOCKS,THREADS_PER_BLOCK>>>(drx,dpo,dims.Nx-1,dims.Ny);    					// add grad(p) to rhs of Ax=b
			bpxOutlet<<<1,26>>>(drx,200); 														// set rhs back to zero at outlet
			AXPY<<<BLOCKS,THREADS_PER_BLOCK>>>(drx,dqx,(T)-1.,(T)1.,dims.Nx-1,dims.Ny);   		// r = r - q
			SpMV<<<BLOCKS,THREADS_PER_BLOCK>>>(dzx,dkuxc,dkuxs,dkuxw,drx,dims.Nx-1,dims.Ny);	// z = M^(-1)r
			DOTGPU<T,128><<<params.blocks,params.blockSize,params.blockSize*sizeof(T)>>>(drh, drx, dzx, dims.Nx-1, dims.Ny);
			cudaMemcpy(hrh, drh, params.blocks*sizeof(T), cudaMemcpyDeviceToHost);
			rhNew = dot(hrh,params.blocks);
			//cout << "Ux residual at start: " << rhNew << endl;
			endIter = rhNew * params.maxResU * params.maxResU;
			iter = 0;
			
			while (rhNew > endIter) {
				iter++;
				if (iter==1) {
					cudaMemcpy(dpx, dzx, sizeof(T)*(dims.Nx-1)*(dims.Ny+2),cudaMemcpyDeviceToDevice);
				}
				else {
					bt = rhNew/rhOld;
					AXPY<<<BLOCKS,THREADS_PER_BLOCK>>>(dpx,dzx,(T)1.,bt,dims.Nx-1,dims.Ny);   		// p = z + beta*p	
				}
				SpMV<<<BLOCKS,THREADS_PER_BLOCK>>>(dqx,duxc,duxs,duxw,dpx,dims.Nx-1,dims.Ny);		// q := Aux p
				DOTGPU<T,128><<<params.blocks,params.blockSize,params.blockSize*sizeof(T)>>>(dsg, dpx, dqx, dims.Nx-1, dims.Ny);
				cudaMemcpy(hsg, dsg, params.blocks*sizeof(T), cudaMemcpyDeviceToHost);
				sg = dot(hsg,params.blocks);
				ap = rhNew/sg;	// alpha = rhoNew / sigma
				AXPY<<<BLOCKS,THREADS_PER_BLOCK>>>(drx,dqx,-ap,(T)1.,dims.Nx-1,dims.Ny); 			// r = r - alpha*q
				AXPY2<<<BLOCKS,THREADS_PER_BLOCK>>>(dux,dpx, ap,(T)1.,dims.Nx-1,dims.Ny);  			// x = x + alpha*p; Note: sizeof(dux) != sizeof(dpx)
				SpMV<<<BLOCKS,THREADS_PER_BLOCK>>>(dzx,dkuxc,dkuxs,dkuxw,drx,dims.Nx-1,dims.Ny);	// z = M^(-1)r
				rhOld = rhNew;
				DOTGPU<T,128><<<params.blocks,params.blockSize,params.blockSize*sizeof(T)>>>(drh, drx, dzx, dims.Nx-1, dims.Ny);
				cudaMemcpy(hrh, drh, params.blocks*sizeof(T), cudaMemcpyDeviceToHost);
				rhNew = dot(hrh,params.blocks);
			}
			//cout << "Ux iter number: " << iter << endl;
			// ********** END solve UX ************
				
			// ********** BEGIN solve UY **********
			duToDr<<<BLOCKS,THREADS_PER_BLOCK>>>(dry, duy,dims.Nx,dims.Ny-1);					// dry := duy
			SpMV<<<BLOCKS,THREADS_PER_BLOCK>>>(dqy,duyc,duys,duyw,dry,dims.Nx,dims.Ny-1);		// q := Auy uy 
			b<<<BLOCKS,THREADS_PER_BLOCK>>>(dry,dims.Nx,dims.Ny-1);	
			//byInlet<<<1,10>>>(dry, duy, 50);
			//byOutlet not necessary due to zero gradient condition
			bpy<<<BLOCKS,THREADS_PER_BLOCK>>>(dry,dpo,dims.Nx,dims.Ny-1);						// add grad(p) to rhs of Ax=b
			AXPY<<<BLOCKS,THREADS_PER_BLOCK>>>(dry,dqy,(T)-1.,(T)1.,dims.Nx,dims.Ny-1);   		// r = r - q
			SpMV<<<BLOCKS,THREADS_PER_BLOCK>>>(dzy,dkuyc,dkuys,dkuyw,dry,dims.Nx,dims.Ny-1);	// z = M^(-1)r
			DOTGPU<T,128><<<params.blocks,params.blockSize,params.blockSize*sizeof(T)>>>(drh, dry, dzy, dims.Nx, dims.Ny-1);
			cudaMemcpy(hrh, drh, params.blocks*sizeof(T), cudaMemcpyDeviceToHost);
			rhNew = dot(hrh,params.blocks);
			//cout << "Uy residual at start: " << rhNew << endl;
			endIter = rhNew * params.maxResU * params.maxResU;
			iter = 0;
			
			while (rhNew > endIter) {
				iter++;
				if (iter==1) {
					cudaMemcpy(dpy, dzy, sizeof(T)*dims.Nx*(dims.Ny+1),cudaMemcpyDeviceToDevice);
				}
				else {
					bt = rhNew/rhOld;
					AXPY<<<BLOCKS,THREADS_PER_BLOCK>>>(dpy,dzy,(T)1.,bt,dims.Nx,dims.Ny-1);   		// p = z + beta*p	
				}
				SpMV<<<BLOCKS,THREADS_PER_BLOCK>>>(dqy,duyc,duys,duxw,dpy,dims.Nx,dims.Ny-1);		// q := Auy p
				DOTGPU<T,128><<<params.blocks,params.blockSize,params.blockSize*sizeof(T)>>>(dsg, dpy, dqy, dims.Nx, dims.Ny-1);
				cudaMemcpy(hsg, dsg, params.blocks*sizeof(T), cudaMemcpyDeviceToHost);
				sg = dot(hsg,params.blocks);
				ap = rhNew/sg;	// alpha = rhoNew / sigma
				AXPY<<<BLOCKS,THREADS_PER_BLOCK>>>(dry,dqy,-ap,(T)1.,dims.Nx,dims.Ny-1);  			// r = r - alpha*q
				AXPY2<<<BLOCKS,THREADS_PER_BLOCK>>>(duy,dpy, ap,(T)1.,dims.Nx,dims.Ny-1);  			// x = x + alpha*p; Note: sizeof(duy) != sizeof(dpy)
				SpMV<<<BLOCKS,THREADS_PER_BLOCK>>>(dzy,dkuyc,dkuys,dkuyw,dry,dims.Nx,dims.Ny-1);	// z = M^(-1)r
				rhOld = rhNew;
				DOTGPU<T,128><<<params.blocks,params.blockSize,params.blockSize*sizeof(T)>>>(drh, dry, dzy, dims.Nx, dims.Ny-1);
				cudaMemcpy(hrh, drh, params.blocks*sizeof(T), cudaMemcpyDeviceToHost);
				rhNew = dot(hrh,params.blocks);
			}
			//cout << "Uy iter number: " << iter << endl;
			// ********** END solve UY ************
			
			bcVelOutlet<<<1,10>>>(dux, duy, 200);
			
			
			
			// ********** BEGIN solve P ***********
			// The finite volume method in computational fluid dynamics, F. Moukalled, L. Mangani, M. Darwish
			// Patankar's SIMPLE
			cudaMemcpy(drp, dp, sizeof(T)*(dims.Nx*dims.Ny+2*dims.Nx),cudaMemcpyDeviceToDevice);
			SpMV<<<BLOCKS,THREADS_PER_BLOCK>>>(dqp,dpc,dps,dpw,drp,dims.Nx,dims.Ny);			// q := Ap p 
			bp<<<BLOCKS,THREADS_PER_BLOCK>>>(drp,dux,duy,dims.Nx,dims.Ny);						// should become at convergence == zero correction field
						
			DOTGPU<T,128><<<params.blocks,params.blockSize,params.blockSize*sizeof(T)>>>(drh, drp, drp, dims.Nx, dims.Ny);
			cudaMemcpy(hrh, drh, params.blocks*sizeof(T), cudaMemcpyDeviceToHost);
			rhNewSIMPLE = dot(hrh,params.blocks);
						
			AXPY<<<BLOCKS,THREADS_PER_BLOCK>>>(drp,dqp,(T)-1.,(T)1.,dims.Nx,dims.Ny);   		// r = r - q
			SpMV<<<BLOCKS,THREADS_PER_BLOCK>>>(dzp,dkpc,dkps,dkpw,drp,dims.Nx,dims.Ny);			// z = M^(-1)r
			DOTGPU<T,128><<<params.blocks,params.blockSize,params.blockSize*sizeof(T)>>>(drh, drp, dzp, dims.Nx, dims.Ny);
			cudaMemcpy(hrh, drh, params.blocks*sizeof(T), cudaMemcpyDeviceToHost);
			rhNew = dot(hrh,params.blocks);
			//cout << "P residual at start: " << rhNew << endl;
			endIter = rhNew * params.maxResP * params.maxResP;
			iter = 0;
			
			while (rhNew > endIter) {
				iter++;
				if (iter==1) {
					cudaMemcpy(dpp, dzp, sizeof(T)*dims.Nx*(dims.Ny+2),cudaMemcpyDeviceToDevice);
				}
				else {
					bt = rhNew/rhOld;
					AXPY<<<BLOCKS,THREADS_PER_BLOCK>>>(dpp,dzp,(T)1.,bt,dims.Nx,dims.Ny);   		// p = z + beta*p	
				}
				SpMV<<<BLOCKS,THREADS_PER_BLOCK>>>(dqp,dpc,dps,dpw,dpp,dims.Nx,dims.Ny);			// q := Ap p
				DOTGPU<T,128><<<params.blocks,params.blockSize,params.blockSize*sizeof(T)>>>(dsg, dpp, dqp, dims.Nx, dims.Ny);
				cudaMemcpy(hsg, dsg, params.blocks*sizeof(T), cudaMemcpyDeviceToHost);
				sg = dot(hsg,params.blocks);
				ap = rhNew/sg;	// alpha = rhoNew / sigma
				AXPY<<<BLOCKS,THREADS_PER_BLOCK>>>(drp,dqp,-ap,(T)1.,dims.Nx,dims.Ny);  			// r = r - alpha*q
				AXPY<<<BLOCKS,THREADS_PER_BLOCK>>>(dp ,dpp, ap,(T)1.,dims.Nx,dims.Ny);  			// x = x + alpha*p
				SpMV<<<BLOCKS,THREADS_PER_BLOCK>>>(dzp,dkpc,dkps,dkpw,drp,dims.Nx,dims.Ny);			// z = M^(-1)r
				rhOld = rhNew;
				DOTGPU<T,128><<<params.blocks,params.blockSize,params.blockSize*sizeof(T)>>>(drh, drp, dzp, dims.Nx, dims.Ny);
				cudaMemcpy(hrh, drh, params.blocks*sizeof(T), cudaMemcpyDeviceToHost);
				rhNew = dot(hrh,params.blocks);
			}
			//cout << "P iter number: " << iter << endl;
			// ********** END solve P ************
			
			
			// ***** BEGIN correct P, UX, UY fields ******
			correctUX<<<BLOCKS,THREADS_PER_BLOCK>>>(dux,dp,dims.Nx-1,dims.Ny); 				// ux = -dt/rho*dp/dx
			correctUY<<<BLOCKS,THREADS_PER_BLOCK>>>(duy,dp,dims.Nx,dims.Ny-1);				// uy = -dt/rho*dp/dy
			AXPY<<<BLOCKS,THREADS_PER_BLOCK>>>(dp,dpo,(T)1.,params.urfP,dims.Nx,dims.Ny);	// p = urfP*p + pold
			cudaMemcpy(dpo, dp, sizeof(T)*dims.Nx*(dims.Ny+2),cudaMemcpyDeviceToDevice);	// pold = p
			cudaMemset(dp ,  0, sizeof(T)*dims.Nx*(dims.Ny+2));
			
			bcVelOutlet<<<1,10>>>(dux, duy, 200);
			// ****** END correct P, UX, UY fields *******
			
			
			// ***** BEGIN check mass conservation *****
			bp<<<BLOCKS,THREADS_PER_BLOCK>>>(dm,dux,duy,dims.Nx,dims.Ny);
			cudaMemcpy(m, dm, sizeof(T)*dims.Nx*(dims.Ny+2), cudaMemcpyDeviceToHost);
			// ****** END check mass conservation ******
			
			
			
			
			
			
		
		}
		// ************** END SIMPLE *****************
		
		cudaMemcpy(duxo, dux, sizeof(T)*(dims.Nx+2)*(dims.Ny+2), cudaMemcpyDeviceToDevice);
		cudaMemcpy(duyo, duy, sizeof(T)*(dims.Nx+2)*(dims.Ny+2), cudaMemcpyDeviceToDevice);
		
		
		// copy back to host and save
		cudaMemcpy(ux, dux, sizeof(T)*(dims.Nx+2)*(dims.Ny+2), cudaMemcpyDeviceToHost);
		cudaMemcpy(uy, duy, sizeof(T)*(dims.Nx+2)*(dims.Ny+2), cudaMemcpyDeviceToHost);
		cudaMemcpy(p , dpo, sizeof(T)*dims.Nx    *(dims.Ny+2), cudaMemcpyDeviceToHost);
		saveDataInTime(ux, uy, p, m, (T)miter, "testTundish");
		
		
		cout << "SIMPLE iter number: " << iterSIMPLE << endl;
	}
	cout << "simulation finished." << endl;
	
	
	
	
	/*// copy back to host and save
	cudaMemcpy(ux, dux, sizeof(T)*(dims.Nx+2)*(dims.Ny+2), cudaMemcpyDeviceToHost);
	cudaMemcpy(uy, duy, sizeof(T)*(dims.Nx+2)*(dims.Ny+2), cudaMemcpyDeviceToHost);
	cudaMemcpy(p , dpo, sizeof(T)*dims.Nx    *(dims.Ny+2), cudaMemcpyDeviceToHost);
	saveDataInTime(ux, uy, p, m, (T)1, "testTundish1e-4");*/
	
	
	
	
	
	
	
	
	
	
		
	
	
	
	
	/*cudaMemcpy(p, drp, sizeof(T)*(dims.Nx)*(dims.Ny+2), cudaMemcpyDeviceToHost);
				ofstream File;
				File.open("ckeck");
				for (int j=0;j<(dims.Ny+2);j++) {
					for (int i=0;i<(dims.Nx);i++) {
						File << p[i+j*(dims.Nx)] << ",";
					}
					File << endl;
				}
				File.close();*/
	
	
	
	
	
	
		
	
	
	
	
	
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	cout<< "ellapsed time (cuda): " << elapsedTime	<< " miliseconds" << endl;
	
	
	
	
			
	cpuFinalize(ux, uy, p, m, hrh, hsg);
	cudaFinalize(dux, duy, dp, dm, duxo, duyo, dpo,
			     duxc, duxs, duxw, dkuxc, dkuxs, dkuxw, 	// Aux
				 drx, dqx, dzx, dpx,						// Aux
				 duyc, duys, duyw, dkuyc, dkuys, dkuyw, 	// Auy
				 dry, dqy, dzy, dpy,						// Auy
				 dpc, dps, dpw, dkpc, dkps, dkpw, 			// Ap
				 drp, dqp, dzp, dpp,						// Ap
	             drh, dsg);
	
	
	return 0;
}
