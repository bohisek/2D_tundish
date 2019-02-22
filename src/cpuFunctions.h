/*
 * cpuFunctions.h
 *
 *  Created on: Jan 23, 2019
 *      Author: jbohacek
 */

#ifndef CPUFUNCTIONS_H_
#define CPUFUNCTIONS_H_


/*// Aux matrix
T *uxc;		// main diagonal ux
T *uxw;		// west flux
T *uxn;		// north flux

// Auy matrix
T *uyc;		// main diagonal uy
T *uyw;		// west flux
T *uyn;		// north flux

// Ap matrix
T *pc;		// main diagonal p
T *pw;		// west flux
T *pn;		// north flux*/

// initialize CPU fields
template <class T>
void cpuInit(T *&ux, T *&uy, T *&p, T *&m,
		     T *&hrh, T *&hsg)
{
	int Nx = dims.Nx;
	int Ny = dims.Ny;
	int blocks = params.blocks;

	ux = new T[(Nx+2)*(Ny+2)];					// MINUS boundary cells and ghost cells i.e. NEITHER storing boundary cells NOR ghost cells
	uy = new T[(Nx+2)*(Ny+2)];						// -||-
	p  = new T[Nx*Ny+2*Nx];
	m  = new T[Nx*Ny+2*Nx];
	hrh = new T[blocks];
	hsg = new T[blocks];


	memset(ux, 0, (Nx+2)*(Ny+2)*sizeof(T));	// set to zero
	memset(uy, 0, (Nx+2)*(Ny+2)*sizeof(T));
	memset(p,  0, (Nx*Ny+2*Nx)*sizeof(T));
	memset(m,  0, (Nx*Ny+2*Nx)*sizeof(T));
	memset(hrh,0, blocks*sizeof(T));
	memset(hsg,0, blocks*sizeof(T));
}


// free memory
template <class T>
void cpuFinalize(T *&ux, T *&uy, T *&p, T *&m,
		         T *&hrh, T *&hsg)
{
	delete[] ux;
	delete[] uy;
	delete[] p;
	delete[] m;
	delete[] hrh;
	delete[] hsg;
}

// save data in time
template <class T>
void saveDataInTime(T *ux,
		T *uy,
		T *p,
		T *m,
		const T t,
		const string name)
{
	int Nx = dims.Nx;
	int Ny = dims.Ny;
	T dx = dims.dx;
	ofstream File;
	stringstream fileName;
	fileName << name << "-" << fixed << (int)t << ".vtk";
	File.open(fileName.str().c_str());
	File << "# vtk DataFile Version 3.0" << endl << "vtk output" << endl;
	File << "ASCII" << endl << "DATASET STRUCTURED_POINTS" << endl;
	File << "DIMENSIONS " << Nx << " " << Ny << " 1" << endl;
	File << "ORIGIN " << 0.5*dx << " " << 0.5*dx << " 1" << endl;
	File << "SPACING " << dx << " " << dx << " " << dx << endl;
	File << "POINT_DATA " << Nx*Ny << endl;
	File << "VECTORS " << "velocity" << " float" << endl;
	for (int i=0; i<(Nx+2)*(Ny+2); ++i) {
		int px = i % (Nx+2);
		int py = i / (Nx+2);
		if ((px<Nx) && (py<Ny))
		{
		File << 0.5 * (ux[i+Nx+2]+ux[i+Nx+3]) << " "			// average from two x-velocities at faces
		     << 0.5 * (uy[i+1]   +uy[i+Nx+3]) << " 0" << endl;	//                  y-velocities
		}
	}
	File << "SCALARS " << "pressure" << " float" << endl;
	File << "LOOKUP_TABLE default" << endl;
	for (int i=0; i<Nx*Ny; ++i) {
		File << p[i+Nx] <<endl;
	}
	File << "SCALARS " << "mass" << " float" << endl;
	File << "LOOKUP_TABLE default" << endl;
	for (int i=0; i<Nx*Ny; ++i) {
		File << m[i+Nx] <<endl;
	}
	File.close();
	cout << "saving VTK (" << fileName.str() << ") at t = " << (int)t << " sec." << endl;
}

// finalize dot product on cpu
T dot(const T *x,
	  const int blocks)
{
	T y = 0;
	for (int i=0; i<blocks; i++) {
		y += x[i];
	}
	return y;
}

#endif /* CPUFUNCTIONS_H_ */
