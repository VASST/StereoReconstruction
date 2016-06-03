/*==========================================================================

  Copyright (c) 2016 Uditha L. Jayarathne, ujayarat@robarts.ca

  Use, modification and redistribution of the software, in source or
  binary forms, are permitted provided that the following terms and
  conditions are met:

  1) Redistribution of the source code, in verbatim or modified
  form, must retain the above copyright notice, this license,
  the following disclaimer, and any notices that refer to this
  license and/or the following disclaimer.

  2) Redistribution in binary form must include the above copyright
  notice, a copy of this license and the following disclaimer
  in the documentation or with other materials provided with the
  distribution.

  3) Modified copies of the source code must be clearly marked as such,
  and must not be misrepresented as verbatim copies of the source code.

  THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES PROVIDE THE SOFTWARE "AS IS"
  WITHOUT EXPRESSED OR IMPLIED WARRANTY INCLUDING, BUT NOT LIMITED TO,
  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
  PURPOSE.  IN NO EVENT SHALL ANY COPYRIGHT HOLDER OR OTHER PARTY WHO MAY
  MODIFY AND/OR REDISTRIBUTE THE SOFTWARE UNDER THE TERMS OF THIS LICENSE
  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, LOSS OF DATA OR DATA BECOMING INACCURATE
  OR LOSS OF PROFIT OR BUSINESS INTERRUPTION) ARISING IN ANY WAY OUT OF
  THE USE OR INABILITY TO USE THE SOFTWARE, EVEN IF ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGES.
  =========================================================================*/

/*! \brief Performs computation of the derivative along x direction. 
*  \param[in] p_in input array of float elements.
*  \param[out] out output (derivative) array of `float` elements.
*/
kernel
void x_gradient (global float *p_in, global float *p_out)
{
    // Workspace dimensions
 /*   int gXdim = get_global_size (0);
    int gYdim = get_global_size (1);

    // Workspace indices
    int gX = get_global_id (0);
    int gY = get_global_id (1);

    // Filter window coordinates
	int lidx = gX-1;
	int ridx = gX+1;
    int outOfBounds = isless ( lidx, 0.f) | isless ( ridx , gXdim);

	p_out[ gY*gXdim + gX ] = select( .5f*p_in[ gY*gXdim + lidx ] + .5f*p_in[ gY*gXdim + ridx], 0.f, outofBouds ); */
}

