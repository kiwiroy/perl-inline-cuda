#ifndef _PERL_CUDA_BRIDGE_H
#define _PERL_CUDA_BRIDGE_H

/* our $VERSION = 0.05; */

int inline_cuda_matrix_multiply(
	SV *perl_A,
	SV *perl_B,
	SV *perl_R,
	int noisy
);

#endif
