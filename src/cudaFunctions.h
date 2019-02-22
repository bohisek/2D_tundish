/*
 * cudaFunctions.h
 *
 *  Created on: Jan 23, 2019
 *      Author: jbohacek
 */

#ifndef CUDAFUNCTIONS_H_
#define CUDAFUNCTIONS_H_

// initialize CUDA fields
template <class T>
void cudaInit(T *&dux, T *&duy, T *&dp, T *&dm, T *&duxo, T *&duyo, T *&dpo,
		      T *&duxc, T *&duxs, T *&duxw, T *&dkuxc, T *&dkuxs, T *&dkuxw, 	// Aux
		      T *&drx, T *&dqx, T *&dzx, T *&dpx,								// Aux
		      T *&duyc, T *&duys, T *&duyw, T *&dkuyc, T *&dkuys, T *&dkuyw, 	// Auy
		      T *&dry, T *&dqy, T *&dzy, T *&dpy,								// Auy
		      T *&dpc, T *&dps, T *&dpw, T *&dkpc, T *&dkps, T *&dkpw, 			// Ap
		      T *&drp, T *&dqp, T *&dzp, T *&dpp,								// Ap
		      T *&drh, T *&dsg)
{
	cudaMemcpyToSymbol(d_dims,   &dims,  sizeof(Dimensions));
	cudaMemcpyToSymbol(d_params, &params,sizeof(Parameters));
	cudaMemcpyToSymbol(d_liquid, &liquid,sizeof(MaterialProperties));

	int Nx = dims.Nx;
	int Ny = dims.Ny;
	int blocks = params.blocks;

	cudaMalloc((void**)&dux , sizeof(T)*(Nx+2)*(Ny+2)); 		// Aux
	cudaMalloc((void**)&duxo, sizeof(T)*(Nx+2)*(Ny+2));			// old values
	cudaMalloc((void**)&drx,  sizeof(T)*((Nx-1)*Ny + 2*(Nx-1)));
	cudaMalloc((void**)&dqx,  sizeof(T)*((Nx-1)*Ny + 2*(Nx-1)));
	cudaMalloc((void**)&dzx,  sizeof(T)*((Nx-1)*Ny + 2*(Nx-1)));
	cudaMalloc((void**)&dpx,  sizeof(T)*((Nx-1)*Ny + 2*(Nx-1)));
	cudaMalloc((void**)&duxc, sizeof(T)*((Nx-1)*Ny + 2*(Nx-1)));
	cudaMalloc((void**)&duxs, sizeof(T)*((Nx-1)*Ny +   (Nx-1)));
	cudaMalloc((void**)&duxw, sizeof(T)*((Nx-1)*Ny + 1       ));
	cudaMalloc((void**)&dkuxc,sizeof(T)*((Nx-1)*Ny + 2*(Nx-1)));
	cudaMalloc((void**)&dkuxs,sizeof(T)*((Nx-1)*Ny +   (Nx-1)));
	cudaMalloc((void**)&dkuxw,sizeof(T)*((Nx-1)*Ny + 1       ));

	cudaMalloc((void**)&duy , sizeof(T)*(Nx+2)*(Ny+2)); 		// Auy
	cudaMalloc((void**)&duyo, sizeof(T)*(Nx+2)*(Ny+2));			// old values
	cudaMalloc((void**)&dry,  sizeof(T)*((Ny-1)*Nx + 2*Nx));
	cudaMalloc((void**)&dqy,  sizeof(T)*((Ny-1)*Nx + 2*Nx));
	cudaMalloc((void**)&dzy,  sizeof(T)*((Ny-1)*Nx + 2*Nx));
	cudaMalloc((void**)&dpy,  sizeof(T)*((Ny-1)*Nx + 2*Nx));
	cudaMalloc((void**)&duyc, sizeof(T)*((Ny-1)*Nx + 2*Nx));
	cudaMalloc((void**)&duys, sizeof(T)*((Ny-1)*Nx +   Nx));
	cudaMalloc((void**)&duyw, sizeof(T)*((Ny-1)*Nx + 1   ));
	cudaMalloc((void**)&dkuyc,sizeof(T)*((Ny-1)*Nx + 2*Nx));
	cudaMalloc((void**)&dkuys,sizeof(T)*((Ny-1)*Nx +   Nx));
	cudaMalloc((void**)&dkuyw,sizeof(T)*((Ny-1)*Nx + 1   ));

	cudaMalloc((void**)&dp , sizeof(T)*(Nx*Ny + 2*Nx));  		// Aup
	cudaMalloc((void**)&dpo, sizeof(T)*(Nx*Ny + 2*Nx));
	cudaMalloc((void**)&drp, sizeof(T)*(Nx*Ny + 2*Nx));
	cudaMalloc((void**)&dqp, sizeof(T)*(Nx*Ny + 2*Nx));
	cudaMalloc((void**)&dzp, sizeof(T)*(Nx*Ny + 2*Nx));
	cudaMalloc((void**)&dpp, sizeof(T)*(Nx*Ny + 2*Nx));
	cudaMalloc((void**)&dpc, sizeof(T)*(Nx*Ny + 2*Nx));
	cudaMalloc((void**)&dps, sizeof(T)*(Nx*Ny +   Nx));
	cudaMalloc((void**)&dpw, sizeof(T)*(Nx*Ny +    1));
	cudaMalloc((void**)&dkpc,sizeof(T)*(Nx*Ny + 2*Nx));
	cudaMalloc((void**)&dkps,sizeof(T)*(Nx*Ny +   Nx));
	cudaMalloc((void**)&dkpw,sizeof(T)*(Nx*Ny +    1));

	cudaMalloc((void**)&drh,  sizeof(T)*blocks      		  );
	cudaMalloc((void**)&dsg,  sizeof(T)*blocks                );

	cudaMalloc((void**)&dm , sizeof(T)*(Nx*Ny + 2*Nx));

	cudaMemset(dux ,0,sizeof(T)*(Nx+2)*(Ny+2)); 			// Aux
	cudaMemset(duxo,0,sizeof(T)*(Nx+2)*(Ny+2));
    cudaMemset(drx  ,0,sizeof(T)*((Nx-1)*Ny + 2*(Nx-1)));
    cudaMemset(dqx  ,0,sizeof(T)*((Nx-1)*Ny + 2*(Nx-1)));
    cudaMemset(dzx  ,0,sizeof(T)*((Nx-1)*Ny + 2*(Nx-1)));
    cudaMemset(dpx  ,0,sizeof(T)*((Nx-1)*Ny + 2*(Nx-1)));
    cudaMemset(duxc ,0,sizeof(T)*((Nx-1)*Ny + 2*(Nx-1)));
    cudaMemset(duxs ,0,sizeof(T)*((Nx-1)*Ny +   (Nx-1)));
    cudaMemset(duxw ,0,sizeof(T)*((Nx-1)*Ny + 1       ));
    cudaMemset(dkuxc,0,sizeof(T)*((Nx-1)*Ny + 2*(Nx-1)));
    cudaMemset(dkuxs,0,sizeof(T)*((Nx-1)*Ny +   (Nx-1)));
    cudaMemset(dkuxw,0,sizeof(T)*((Nx-1)*Ny + 1       ));

    cudaMemset(duy , 0,sizeof(T)*(Nx+2)*(Ny+2));			// Auy
	cudaMemset(duyo, 0,sizeof(T)*(Nx+2)*(Ny+2));
    cudaMemset(dry,  0,sizeof(T)*((Ny-1)*Nx + 2*Nx));
	cudaMemset(dqy,  0,sizeof(T)*((Ny-1)*Nx + 2*Nx));
	cudaMemset(dzy,  0,sizeof(T)*((Ny-1)*Nx + 2*Nx));
	cudaMemset(dpy,  0,sizeof(T)*((Ny-1)*Nx + 2*Nx));
	cudaMemset(duyc, 0,sizeof(T)*((Ny-1)*Nx + 2*Nx));
	cudaMemset(duys, 0,sizeof(T)*((Ny-1)*Nx +   Nx));
	cudaMemset(duyw, 0,sizeof(T)*((Ny-1)*Nx + 1   ));
	cudaMemset(dkuyc,0,sizeof(T)*((Ny-1)*Nx + 2*Nx));
	cudaMemset(dkuys,0,sizeof(T)*((Ny-1)*Nx +   Nx));
	cudaMemset(dkuyw,0,sizeof(T)*((Ny-1)*Nx + 1   ));

	cudaMemset(dp  ,0,sizeof(T)*(Nx*Ny + 2*Nx));   			// Aup
	cudaMemset(dpo ,0,sizeof(T)*(Nx*Ny + 2*Nx));
	cudaMemset(drp, 0,sizeof(T)*(Nx*Ny + 2*Nx));
	cudaMemset(dqp, 0,sizeof(T)*(Nx*Ny + 2*Nx));
	cudaMemset(dzp, 0,sizeof(T)*(Nx*Ny + 2*Nx));
	cudaMemset(dpp, 0,sizeof(T)*(Nx*Ny + 2*Nx));
	cudaMemset(dpc, 0,sizeof(T)*(Nx*Ny + 2*Nx));
	cudaMemset(dps, 0,sizeof(T)*(Nx*Ny +   Nx));
	cudaMemset(dpw, 0,sizeof(T)*(Nx*Ny +    1));
	cudaMemset(dkpc,0,sizeof(T)*(Nx*Ny + 2*Nx));
	cudaMemset(dkps,0,sizeof(T)*(Nx*Ny +   Nx));
	cudaMemset(dkpw,0,sizeof(T)*(Nx*Ny +    1));

    cudaMemset(drh,  0,sizeof(T)*blocks      		   );
    cudaMemset(dsg,  0,sizeof(T)*blocks                );

    cudaMemset(dm  ,0,sizeof(T)*(Nx*Ny + 2*Nx));

}


// destroy CUDA fields
void cudaFinalize(T *&dux, T *&duy, T *&dp, T *&dm, T *&duxo, T *&duyo, T *&dpo,
				  T *&duxc, T *&duxs, T *&duxw, T *&dkuxc, T *&dkuxs, T *&dkuxw, 	// Aux
				  T *&drx, T *&dqx, T *&dzx, T *&dpx,								// Aux
				  T *&duyc, T *&duys, T *&duyw, T *&dkuyc, T *&dkuys, T *&dkuyw, 	// Auy
				  T *&dry, T *&dqy, T *&dzy, T *&dpy,								// Auy
				  T *&dpc, T *&dps, T *&dpw, T *&dkpc, T *&dkps, T *&dkpw, 			// Ap
				  T *&drp, T *&dqp, T *&dzp, T *&dpp,								// Ap
				  T *&drh, T *&dsg)
{
	cudaFree(dux);
	cudaFree(duxo);
	cudaFree(duxc);	// Aux
	cudaFree(duxs);
	cudaFree(duxw);
	cudaFree(dkuxc);
	cudaFree(dkuxs);
	cudaFree(dkuxw);
	cudaFree(drx);
	cudaFree(dqx);
	cudaFree(dzx);
	cudaFree(dpx);

	cudaFree(duy);
	cudaFree(duyo);
	cudaFree(duyc);	// Auy
	cudaFree(duys);
	cudaFree(duyw);
	cudaFree(dkuyc);
	cudaFree(dkuys);
	cudaFree(dkuyw);
	cudaFree(dry);
	cudaFree(dqy);
	cudaFree(dzy);
	cudaFree(dpy);

	cudaFree(dp); 	// Aup
	cudaFree(dpc);
	cudaFree(dps);
	cudaFree(dpw);
	cudaFree(dkpc);
	cudaFree(dkps);
	cudaFree(dkpw);
	cudaFree(drp);
	cudaFree(dqp);
	cudaFree(dzp);
	cudaFree(dpp);

	cudaFree(dm);

	cudaFree(drh);
	cudaFree(dsg);

	cudaDeviceReset();
}

// patch anything to dux
template <class T>
__global__ void patchDux(T *dux)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int Nx = d_dims.Nx;
	int Ny = d_dims.Ny;
	int px = i % (Nx+2);
	int py = i / (Nx+2);

	if (i<(Nx+2)*(Ny+2)) {
		if ((px>150) && (px<175) && (py>35) && (py<45)) {
			dux[i+Nx+2] = 1.5;
		}
	}
}

// patch anything to duy
template <class T>
__global__ void patchDuy(T *duy)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int Nx = d_dims.Nx;
	int Ny = d_dims.Ny;
	int px = i % (Nx+2);
	int py = i / (Nx+2);

	if (i<(Nx+2)*(Ny+2)) {
		if ((px>50) && (px<75) && (py>15) && (py<35)) {
			duy[i+Nx+2] = -0.5;
		}
	}
}


// advection of ux
template <class T>
__global__ void advectUx(T *dux, T *duxold, T *duyold)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int Nx = d_dims.Nx;
	int Ny = d_dims.Ny;
	T dx = d_dims.dx;
	T dt = d_params.dt;
	int px = i % (Nx+2);
	int py = i / (Nx+2);
	int px0, py0, px1, py1;
	T x, y, dx0, dx1, dy0, dy1;

	// skip boundary values
	if ((px>0) && (py>0) && (px<Nx) && (py<Ny+1)) { // skip left and right boundary
		// move "backwards" in time
		x = px - dt * duxold[i] / dx;
		y = py - dt * 0.25 * (duyold[i] + duyold[i+1] + duyold[i-Nx-2] + duyold[i-Nx-1]) / dx;

		// if the velocity goes over the boundary, clamp it
		if (x<0) 	x=0; 	if (x>Nx)  		x=Nx;
		if (y<0.5) 	y=0.5; 	if (y>Ny+0.5) 	y=Ny+0.5;

		// setup bilinear interpolation "corner points"
		px0 = (int) x; px1 = px0 + 1; dx1 = x - px0; dx0 = 1 - dx1;
		py0 = (int) y; py1 = py0 + 1; dy1 = y - py0; dy0 = 1 - dy1;

		// perform a bilinear interpolation
		dux[i] = dx0 * (dy0 * duxold[px0 + (Nx+2)*py0] + dy1 * duxold[px0 + (Nx+2)*py1]) +
				 dx1 * (dy0 * duxold[px1 + (Nx+2)*py0] + dy1 * duxold[px1 + (Nx+2)*py1]);
	}
}

// advection of uy
template <class T>
__global__ void advectUy(T *duy, T *duxold, T *duyold)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int Nx = d_dims.Nx;
	int Ny = d_dims.Ny;
	T dx = d_dims.dx;
	T dt = d_params.dt;
	int px = i % (Nx+2);
	int py = i / (Nx+2);
	int px0, py0, px1, py1;
	T x, y, dx0, dx1, dy0, dy1;

	// skip boundary values
	if ((px>0) && (py>0) && (px<Nx+1) && (py<Ny)) { // skip left and right boundary
		// move "backwards" in time
		x = px - dt * 0.25 * (duxold[i] + duxold[i-1] + duxold[i+Nx+2] + duxold[i+Nx+1]) / dx;
		y = py - dt * duyold[i] / dx;

		// if the velocity goes over the boundary, clamp it
		if (x<0.5) 	x=0.5; 	if (x>Nx+0.5)	x=Nx+0.5;
		if (y<0) 	y=0; 	if (y>Ny) 		y=Ny;

		// setup bilinear interpolation "corner points"
		px0 = (int) x; px1 = px0 + 1; dx1 = x - px0; dx0 = 1 - dx1;
		py0 = (int) y; py1 = py0 + 1; dy1 = y - py0; dy0 = 1 - dy1;

		// perform a bilinear interpolation
		duy[i] = dx0 * (dy0 * duyold[px0 + (Nx+2)*py0] + dy1 * duyold[px0 + (Nx+2)*py1]) +
				 dx1 * (dy0 * duyold[px1 + (Nx+2)*py0] + dy1 * duyold[px1 + (Nx+2)*py1]);
	}
}


// NO slip wall for velocity
template <class T>
__global__ void bcVelWallNoslip(T *dux, T *duy)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int Nx = d_dims.Nx;
	int Ny = d_dims.Ny;
	int px = i % (Nx+2);
	int py = i / (Nx+2);

	// Skip Inner Values
	if(px == 0) {
		dux[i] = 0;
		duy[i] = -duy[i+1];
	}
	else if(py == 0) {
		dux[i] = -dux[i+Nx+2];
		duy[i] = 0;
	}
	else if(px == Nx) {
		dux[i]   = 0;
		dux[i+1] = 0; // doesn't need to be set <= not used in simulation
		duy[i+1] = -duy[i];
	}

	else if(py == Ny) {
		dux[i+Nx+2] = -dux[i];
		duy[i]      = 0;
		duy[i+Nx+2] = 0; // doesn't need to be set <= not used in simulation
	}
}

// slip wall for velocity
template <class T>
__global__ void bcVelWallSlip(T *dux, T *duy)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int Nx = d_dims.Nx;
	int Ny = d_dims.Ny;
	int px = i % (Nx+2);
	int py = i / (Nx+2);

	// Skip Inner Values
	if(px == 0) {
		dux[i] = 0;
		duy[i] = duy[i+1];
	}
	else if(py == 0) {
		dux[i] = dux[i+Nx+2];
		duy[i] = 0;
	}
	else if(px == Nx) {
		dux[i]   = 0;
		dux[i+1] = 0; // doesn't need to be set <= not used in simulation
		duy[i+1] = duy[i];
	}
	else if(py == Ny) {
		dux[i+Nx+2] = dux[i];
		duy[i]      = 0;
		duy[i+Nx+2] = 0; // doesn't need to be set <= not used in simulation
	}
}

// velocity inlet (shroud)
template <class T>
__global__ void bcVelInlet(T *dux, T *duy, const T UY, const int start)
{
	int i  = threadIdx.x + blockIdx.x * blockDim.x;
	int Nx = d_dims.Nx;
	int Ny = d_dims.Ny;

	duy[i+start+1+(Nx+2)*Ny] = UY; 										// normal velocity
	if (i>0)	dux[i+start+(Nx+2)*(Ny+1)] = -dux[i+start+(Nx+2)*Ny];	// zero tangential velocity
}

// velocity outlet (sen)
template <class T>
__global__ void bcVelOutlet(T *dux, T *duy, const int start)
{
	int i  = threadIdx.x + blockIdx.x * blockDim.x;
	int Nx = d_dims.Nx;

	duy[i+start+1] = duy[i+start+Nx+3] + dux[i+start+Nx+3] - dux[i+start+Nx+2]; 			// fullfil continuity
	if (i>0)	dux[i+start] = dux[i+start+Nx+2];	// zero gradient
}


// fill diagonals of Aux
template <class T>
__global__ void Aux(T *duxc, T *duxs, T *duxw)	// launch (Nx-1)*Ny threads
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx-1;
	int ny = d_dims.Ny;

	T ac   = d_params.ac;
	T nu   = d_liquid.nu;
	T urf  = d_params.urfU;

	int px = i % nx;
	int py = i / nx;

	if (i<nx*ny)
	{
		duxc[i+nx] = ac/urf + 4*nu;
		duxw[i] = -nu;
		duxs[i] = -nu;
		if (px==0)	duxw[i] = 0;

		if (py==0) {
			duxc[i+nx] += nu;
			duxs[i] = 0;
		}

		if (py==ny-1)	duxc[i+nx] += nu;
	}
}

// fill diagonals of Auy
template <class T>
__global__ void Auy(T *duyc, T *duys, T *duyw)	// launch Nx*(Ny-1) threads
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx;
	int ny = d_dims.Ny-1;

	T ac   = d_params.ac;
	T nu   = d_liquid.nu;
	T urf  = d_params.urfU;

	int px = i % nx;
	int py = i / nx;

	if (i<nx*ny)
	{
		duyc[i+nx] = ac/urf + 4*nu;
		duyw[i] = -nu;
		duys[i] = -nu;

		if (px==0) {
			duyc[i+nx] += nu;
			duyw[i] = 0;
		}

		if (px==nx-1)	duyc[i+nx] += nu;

		if (py==0)	duys[i] = 0;
	}
}

// fill diagonals of Ap
template <class T>
__global__ void Ap(T *dpc, T *dps, T *dpw)	// launch Nx*Ny threads
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx;
	int ny = d_dims.Ny;

	T D = d_params.dt / (d_liquid.rho * d_dims.dx);

	int px = i % nx;
	int py = i / nx;

	if (i<nx*ny)
	{
		dpc[i+nx] = 4*D;
		dpw[i] = -D;
		dps[i] = -D;

		if (px==0) {
			dpc[i+nx] -= D;
			dpw[i] = 0;
		}

		if (py==0)	{
			dpc[i+nx] -= D;
			dps[i] = 0;
		}

		if (px==nx-1)	dpc[i+nx] -= D;

		if (py==ny-1)	dpc[i+nx] -= D;
	}
}


// modify diagonals of Aux at outlet
template <class T>
__global__ void AuxOutlet(T *duxc, const int start)	// launch outletWidth number of threads
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx-1;

	T nu   = d_liquid.nu;

	if (i>0) 	duxc[i+start+nx] -= 2*nu;  // no viscous force; zero gradient boundary condition

}

// modify diagonals of Aux at outlet
template <class T>
__global__ void AuyOutlet(T *duyc, const int start)	// launch outletWidth number of threads
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx;

	T nu   = d_liquid.nu;

	duyc[i+start+nx] -= nu; // no viscous force; zero gradient boundary condition
}


// modify diagonals of Ap at outlet
template <class T>
__global__ void ApOutlet(T *dpc, const int start)	// launch outletWidth number of threads
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx;

	T D = d_params.dt / (d_liquid.rho * d_dims.dx);

	dpc[i+start+nx]   = 4*D; // Dirichlet at outlet = zero pressure
}


// modify by at inlet (Note: at inlet bx without change)
template <class T>
__global__ void byInlet(T *dry, const T *duy, const int start)	// launch outletWidth number of threads
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx;
	int ny = d_dims.Ny-1;

	T nu   = d_liquid.nu;

	dry[i+start+nx*ny] += nu * duy[i+start+1+(nx+2)*(ny+2)];
}

// fill bpx add dp/dx to right-handside Aux = bx
template <class T>
__global__ void bpx(T *dr,
		const T *dp,
		const int Nx,
		const int Ny)	// launch (Nx-1)*Ny threads
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx;
	int px = i % Nx;
	int py = i / Nx;
	T rho  = d_liquid.rho;
	T dx   = d_dims.dx;

	if (i<Nx*Ny)	dr[i+Nx] += dx * (dp[px+(py+1)*nx]-dp[px+1+(py+1)*nx]) / rho;  // pw-pe; 1_OK
}

// fill bpx add dp/dx to right-handside Aux = bx
template <class T>
__global__ void bpxOutlet(T *dr, const int start)  // launch outletWidth number of threads + 1
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx-1;

	dr[i+start-1+nx] = 0;
}

// fill bpy add dp/dy to right-handside of Auy = by
template <class T>
__global__ void bpy(T *dr,
		const T *dp,
		const int Nx,
		const int Ny)	// launch Nx*(Ny-1) threads
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx;
	int px = i % Nx;
	int py = i / Nx;
	T rho  = d_liquid.rho;
	T dx   = d_dims.dx;

	if (i<Nx*Ny)	dr[i+Nx] += dx * (dp[px+(py+1)*nx]-dp[px+(py+2)*nx]) / rho;  // pn-ps; 1_OK
}

// update UX field after solving P pressure
template <class T>
__global__ void correctUX(T *dux,
		const T *dp,
		const int Nx,
		const int Ny)	// launch (Nx-1)*Ny threads
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx;
	int px = i % Nx;
	int py = i / Nx;
	T rho  = d_liquid.rho;
	T dt   = d_params.dt;
	T dx   = d_dims.dx;

	if (i<Nx*Ny)	dux[px+1+(py+1)*(nx+2)] -=  dt * (dp[px+1+(py+1)*nx]-dp[px+(py+1)*nx]) / (rho*dx);  // pe-pw; 1_OK
}

// update UY field after solving P pressure
template <class T>
__global__ void correctUY(T *duy,
		const T *dp,
		const int Nx,
		const int Ny)	// launch (Nx-1)*Ny threads
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx;
	int px = i % Nx;
	int py = i / Nx;
	T rho  = d_liquid.rho;
	T dt   = d_params.dt;
	T dx   = d_dims.dx;

	if (i<Nx*Ny)	duy[px+1+(py+1)*(nx+2)] -=  dt * (dp[px+(py+2)*nx]-dp[px+(py+1)*nx]) / (rho*dx);  // pn-ps; 1_OK
}




// **********************************************************
// ****** generic functions for all unknowns ux, uy, p ******
// **********************************************************


// copy dux to drx
template <class T>
__global__ void duToDr(T *dr,
		               const T *du,
		               const int Nx,
		               const int Ny)	// launch (Nx-1)*Ny threads
{
	int i  = threadIdx.x + blockIdx.x * blockDim.x;
	int nx = d_dims.Nx;
	int px = i % Nx;
	int py = i / Nx;

	if (i<Nx*Ny)	dr[i+Nx] = du[px+1+(py+1)*(nx+2)];       // dux[Nx+2 + px+1 + py*(Nx+2)]
}


// fill b (Ax=b); right-handside of Au = b
template <class T>
__global__ void b(T *dr,
		const int Nx,
		const int Ny)	// launch (Nx-1)*Ny threads
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	T ac   = d_params.ac;
	T urf  = d_params.urfU;

	if (i<Nx*Ny)	dr[i+Nx] *= ac/urf;
}

// fill b (Ax=b); right-handside of Au = b
template <class T>
__global__ void bp(T *dr,
		const T *dux,
		const T *duy,
		const int Nx,
		const int Ny)	// launch (Nx-1)*Ny threads
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int px = i % Nx;
	int py = i / Nx;

	if (i<Nx*Ny)	dr[i+Nx] = dux[px+(py+1)*(Nx+2)] - dux[px+1+(py+1)*(Nx+2)]
			                 + duy[px+1+py*(Nx+2)]   - duy[px+1+(py+1)*(Nx+2)]; // 1_OK
}

// SPMV (sparse matrix-vector multiplication)
template <class T>
__global__ void SpMV(T *y,  	// launch Nx*Ny threads
		T *dc,
		T *ds,
		T *dw,
		const T *x,
		const int Nx,
		const int Ny)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid<Nx*Ny) {
		tid += Nx;
		y[tid]  = dc[tid]      * x[tid]     // center
	    	    + ds[tid]      * x[tid+Nx]  // north               N
	            + dw[tid-Nx+1] * x[tid+1]   // east              W C E
	            + ds[tid-Nx]   * x[tid-Nx]  // south               S
	            + dw[tid-Nx]   * x[tid-1];  // west
	}
}

// Truncated Neumann series 1
template <class T>
__global__ void makeTNS1(T *dkc,
		T *dks,
		T *dkw,
		const T *dc,
		const T *ds,
		const T *dw,
		const int Nx,
		const int Ny)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid<Nx*Ny) {
		T tstC1 = 1. / dc[tid+Nx];
		T tstC2 = 0.;
		T tstC3 = 0.;
		if (tid < Nx*Ny-Nx) tstC2 = 1. / dc[tid+2*Nx];
		if (tid < Nx*Ny-1)  tstC3 = 1. / dc[tid+Nx+1];

		dkc[tid+Nx] = tstC1 * ( 1 + ds[tid+Nx] * ds[tid+Nx]  * tstC1 * tstC2  + dw[tid+1] * dw[tid+1]  * tstC1 * tstC3 );

		dks[tid+Nx] = -ds[tid+Nx] * tstC1 * tstC2;

		dkw[tid+1]  = -dw[tid+1]  * tstC1 * tstC3;
    }
}

// AXPY (y := alpha*x + beta*y)
template <class T>
__global__ void AXPY(T *y,	// launch Nx*Ny threads
		const T *x,
		const T alpha,
		const T beta,
		const int Nx,
		const int Ny)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid<Nx*Ny) {
		tid += Nx;
		y[tid] = alpha * x[tid] + beta * y[tid];
	}
}

// AXPY2 (y := alpha*x + beta*y)
// sizeof(dy) != sizeof(x)
template <class T>
__global__ void AXPY2(T *y,	// launch Nx*Ny threads
		const T *x,
		const T alpha,
		const T beta,
		const int Nx,
		const int Ny)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid<Nx*Ny) {
		int nx = d_dims.Nx;
		int px = tid % Nx;
		int py = tid / Nx;
		y[px+1+(py+1)*(nx+2)] = alpha * x[tid+Nx] + beta * y[px+1+(py+1)*(nx+2)]; // 1_OK
	}
}

// DOT PRODUCT
template <class T, unsigned int blockSize>
__global__ void DOTGPU(T *c,
		const T *a,
		const T *b,
		const int Nx,
		const int Ny)
{
	extern __shared__ T cache[];

	unsigned int tid = threadIdx.x;
	unsigned int i = tid + blockIdx.x * (blockSize * 2);
	unsigned int gridSize = (blockSize*2)*gridDim.x;


	cache[tid] = 0;

	while(i<Nx*Ny) {
		cache[tid] += a[i+Nx] * b[i+Nx] + a[i+Nx+blockSize] * b[i+Nx+blockSize];
		i += gridSize;
	}

	__syncthreads();

	if(blockSize >= 512) {	if(tid < 256) { cache[tid] += cache[tid + 256]; } __syncthreads(); }
	if(blockSize >= 256) {	if(tid < 128) { cache[tid] += cache[tid + 128]; } __syncthreads(); }
	if(blockSize >= 128) {	if(tid < 64 ) { cache[tid] += cache[tid + 64 ]; } __syncthreads(); }

	if(tid < 32) {
		cache[tid] += cache[tid + 32];
		cache[tid] += cache[tid + 16];
		cache[tid] += cache[tid + 8];
		cache[tid] += cache[tid + 4];
		cache[tid] += cache[tid + 2];
		cache[tid] += cache[tid + 1];
	}

	if (tid == 0) c[blockIdx.x] = cache[0];
}



#endif /* CUDAFUNCTIONS_H_ */
